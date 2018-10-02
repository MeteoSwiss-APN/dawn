//===--------------------------------------------------------------------------------*- C++ -*-===//
//                          _
//                         | |
//                       __| | __ ___      ___ ___
//                      / _` |/ _` \ \ /\ / / '_  |
//                     | (_| | (_| |\ V  V /| | | |
//                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "dawn/Optimizer/PassFieldVersioning.h"
#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/Extents.h"
#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/AccessComputation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/Renaming.h"
#include "dawn/SIR/AST.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include <iostream>
#include <set>
namespace dawn {

namespace {

/// @brief Register all referenced AccessIDs
struct AccessIDGetter : public ASTVisitorForwarding {
  const iir::IIR* iir_;
  std::set<int> AccessIDs;

  AccessIDGetter(const iir::IIR* iir) : iir_(iir) {}

  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) override {
    AccessIDs.insert(iir_->getMetaData()->getAccessIDFromExpr(expr));
  }
};

/// @brief Compute the AccessIDs of the left and right hand side expression of the assignment
static void getAccessIDFromAssignment(iir::IIR* iir_,
                                      AssignmentExpr* assignment, std::set<int>& LHSAccessIDs,
                                      std::set<int>& RHSAccessIDs) {
  auto computeAccessIDs = [&](const std::shared_ptr<Expr>& expr, std::set<int>& AccessIDs) {
    AccessIDGetter getter{iir_};
    expr->accept(getter);
    AccessIDs = std::move(getter.AccessIDs);
  };

  computeAccessIDs(assignment->getLeft(), LHSAccessIDs);
  computeAccessIDs(assignment->getRight(), RHSAccessIDs);
}

/// @brief Check if the extent is a stencil-extent (i.e non-pointwise in the horizontal and a
/// counter loop acesses in the vertical)
static bool isHorizontalStencilOrCounterLoopOrderExtent(const iir::Extents& extent,
                                                        iir::LoopOrderKind loopOrder) {
  return !extent.isHorizontalPointwise() ||
         extent.getVerticalLoopOrderAccesses(loopOrder).CounterLoopOrder;
}

/// @brief Report a race condition in the given `statement`
static void reportRaceCondition(const Statement& statement,
                                iir::IIR* iir) {
  DiagnosticsBuilder diag(DiagnosticsKind::Error, statement.ASTStmt->getSourceLocation());

  if(isa<IfStmt>(statement.ASTStmt.get())) {
    diag << "unresolvable race-condition in body of if-statement";
  } else {
    diag << "unresolvable race-condition in statement";
  }

  iir->getDiagnostics().report(diag);

  // Print stack trace of stencil calls
  if(statement.StackTrace) {
    std::vector<sir::StencilCall*>& stackTrace = *statement.StackTrace;
    for(int i = stackTrace.size() - 1; i >= 0; --i) {
      DiagnosticsBuilder note(DiagnosticsKind::Note, stackTrace[i]->Loc);
      note << "detected during instantiation of stencil-call '" << stackTrace[i]->Callee << "'";
      iir->getDiagnostics().report(note);
    }
  }
}

} // anonymous namespace

PassFieldVersioning::PassFieldVersioning() : Pass("PassFieldVersioning", true), numRenames_(0) {}

bool PassFieldVersioning::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  numRenames_ = 0;

  for(const auto& stencilPtr : stencilInstantiation->getIIR()->getChildren()) {
    iir::Stencil& stencil = *stencilPtr;

    // Iterate multi-stages backwards
    int stageIdx = stencil.getNumStages() - 1;
    for(auto multiStageRit = stencil.childrenRBegin(), multiStageRend = stencil.childrenREnd();
        multiStageRit != multiStageRend; ++multiStageRit) {
      iir::MultiStage& multiStage = (**multiStageRit);
      iir::LoopOrderKind loopOrder = multiStage.getLoopOrder();

      std::shared_ptr<iir::DependencyGraphAccesses> newGraph, oldGraph;
      newGraph =
          std::make_shared<iir::DependencyGraphAccesses>(stencilInstantiation->getIIR().get());

      // Iterate stages bottom -> top
      for(auto stageRit = multiStage.childrenRBegin(), stageRend = multiStage.childrenREnd();
          stageRit != stageRend; ++stageRit) {
        iir::Stage& stage = (**stageRit);
        iir::DoMethod& doMethod = stage.getSingleDoMethod();

        // Iterate statements bottom -> top
        for(int stmtIndex = doMethod.getChildren().size() - 1; stmtIndex >= 0; --stmtIndex) {
          oldGraph = newGraph->clone();

          auto& stmtAccessesPair = doMethod.getChildren()[stmtIndex];
          newGraph->insertStatementAccessesPair(stmtAccessesPair);

          // Try to resolve race-conditions by using double buffering if necessary
          auto rc = fixRaceCondition(stencilInstantiation->getIIR().get(), newGraph.get(), stencil,
                                     doMethod, loopOrder, stageIdx, stmtIndex);

          if(rc == RCKind::RK_Unresolvable) {
            // Nothing we can do ... bail out
            return false;
          } else if(rc == RCKind::RK_Fixed) {
            // We fixed a race condition (this means some fields have changed and our current graph
            // is invalid)
            newGraph = oldGraph;
            newGraph->insertStatementAccessesPair(stmtAccessesPair);
          }
        }
        stage.update(iir::NodeUpdateType::level);
      }
      stageIdx--;
    }

    for(const auto& ms : iterateIIROver<iir::MultiStage>(stencil)) {
      ms->update(iir::NodeUpdateType::levelAndTreeAbove);
    }
  }

  if(stencilInstantiation->getIIR()->getOptions().ReportPassFieldVersioning && numRenames_ == 0)
    std::cout << "\nPASS: " << getName() << ": " << stencilInstantiation->getIIR()->getMetaData()->getName()
              << ": no rename\n";
  return true;
}

PassFieldVersioning::RCKind
PassFieldVersioning::fixRaceCondition(iir::IIR* iir, const iir::DependencyGraphAccesses* graph,
                                      iir::Stencil& stencil, iir::DoMethod& doMethod,
                                      iir::LoopOrderKind loopOrder, int stageIdx, int index) {
  using Vertex = iir::DependencyGraphAccesses::Vertex;
  using Edge = iir::DependencyGraphAccesses::Edge;

  Statement& statement = *doMethod.getChildren()[index]->getStatement();

  int numRenames = 0;

  // Vector of strongly connected components with atleast one stencil access
  auto stencilSCCs = make_unique<std::vector<std::set<int>>>();

  // Find all strongly connected components in the graph ...
  auto SCCs = make_unique<std::vector<std::set<int>>>();
  graph->findStronglyConnectedComponents(*SCCs);

  // ... and add those which have atleast one stencil access
  for(std::set<int>& scc : *SCCs) {
    bool isStencilSCC = false;

    for(int FromAccessID : scc) {
      std::size_t FromVertexID = graph->getVertexIDFromID(FromAccessID);

      for(const Edge& edge : *graph->getAdjacencyList()[FromVertexID]) {
        if(scc.count(graph->getIDFromVertexID(edge.ToVertexID)) &&
           isHorizontalStencilOrCounterLoopOrderExtent(edge.Data, loopOrder)) {
          isStencilSCC = true;
          stencilSCCs->emplace_back(std::move(scc));
          break;
        }
      }

      if(isStencilSCC)
        break;
    }
  }

  if(stencilSCCs->empty()) {
    // Check if we have self dependencies e.g `u = u(i+1)` (Our SCC algorithm does not capture SCCs
    // of size one i.e single nodes)
    for(const auto& AccessIDVertexPair : graph->getVertices()) {
      const Vertex& vertex = AccessIDVertexPair.second;

      for(const Edge& edge : *graph->getAdjacencyList()[vertex.VertexID]) {
        if(edge.FromVertexID == edge.ToVertexID &&
           isHorizontalStencilOrCounterLoopOrderExtent(edge.Data, loopOrder)) {
          stencilSCCs->emplace_back(std::set<int>{vertex.ID});
          break;
        }
      }
    }
  }

  // If we only have non-stencil SCCs and there are no input and output fields (i.e we don't have a
  // DAG) we have to break (by renaming) one of the SCCs to get a DAG. For example:
  //
  //  field_a = field_b;
  //  field_b = field_a;
  //
  // needs to be renamed to
  //
  //  field_a = field_b_1;
  //  field_b = field_a;
  //
  if(stencilSCCs->empty() && !SCCs->empty() && !graph->isDAG()) {
    stencilSCCs->emplace_back(std::move(SCCs->front()));
  }

  if(stencilSCCs->empty())
    return RCKind::RK_Nothing;

  // Check whether our statement is an `ExprStmt` and contains an `AssignmentExpr`. If not,
  // we cannot perform any double buffering (e.g if there is a problem inside an `IfStmt`, nothing
  // we can do (yet ;))
  AssignmentExpr* assignment = nullptr;
  if(ExprStmt* stmt = dyn_cast<ExprStmt>(statement.ASTStmt.get()))
    assignment = dyn_cast<AssignmentExpr>(stmt->getExpr().get());


  if(!assignment) {
    if(iir->getOptions().DumpRaceConditionGraph)
      graph->toDot("rc_" + iir->getMetaData()->getName() + ".dot");
    reportRaceCondition(statement, iir);
    return RCKind::RK_Unresolvable;
  }

  // Get AccessIDs of the LHS and RHS
  std::set<int> LHSAccessIDs, RHSAccessIDs;
  getAccessIDFromAssignment(iir, assignment, LHSAccessIDs, RHSAccessIDs);

  DAWN_ASSERT_MSG(LHSAccessIDs.size() == 1, "left hand side should only have only one AccessID");
  int LHSAccessID = *LHSAccessIDs.begin();

  // If the LHSAccessID is not part of the SCC, we cannot resolve the race-condition
  for(std::set<int>& scc : *stencilSCCs) {
    if(!scc.count(LHSAccessID)) {
      if(iir->getOptions().DumpRaceConditionGraph)
        graph->toDot("rc_" + iir->getMetaData()->getName() + ".dot");
      reportRaceCondition(statement, iir);
      return RCKind::RK_Unresolvable;
    }
  }

  DAWN_ASSERT_MSG(stencilSCCs->size() == 1, "only one strongly connected component can be handled");
  std::set<int>& stencilSCC = (*stencilSCCs)[0];

  std::set<int> renameCandiates;
  for(int AccessID : stencilSCC) {
    if(RHSAccessIDs.count(AccessID))
      renameCandiates.insert(AccessID);
  }

  if(iir->getOptions().ReportPassFieldVersioning)
    std::cout << "\nPASS: " << getName() << ": " << iir->getMetaData()->getName()
              << ": rename:" << statement.ASTStmt->getSourceLocation().Line;

  // Create a new multi-versioned field and rename all occurences
  for(int oldAccessID : renameCandiates) {
    int newAccessID = createVersionAndRename(iir, oldAccessID, &stencil, stageIdx, index,
                                                           assignment->getRight(),
                                                           RenameDirection::RD_Above);

    if(iir->getOptions().ReportPassFieldVersioning)
      std::cout << (numRenames != 0 ? ", " : " ") << iir->getMetaData()->getNameFromAccessID(oldAccessID)
                << ":" << iir->getMetaData()->getNameFromAccessID(newAccessID);

    numRenames++;
  }

  if(iir->getOptions().ReportPassFieldVersioning && numRenames > 0)
    std::cout << "\n";

  numRenames_ += numRenames;
  return RCKind::RK_Fixed;
}


int createVersionAndRename(iir::IIR* iir_, int AccessID, iir::Stencil* stencil, int curStageIdx,
                           int curStmtIdx, std::shared_ptr<Expr>& expr, RenameDirection dir) {

  int newAccessID = iir_->getMetaData()->nextUID();

  if(iir_->getMetaData()->isField(AccessID)) {
    if(iir_->getMetaData()->isMultiVersionedField(AccessID)) {
      // Field is already multi-versioned, append a new version
      auto versions = iir_->getMetaData()->getVariableVersions().getVersions(AccessID);

      // Set the second to last field to be a temporary (only the first and the last field will be
      // real storages, all other versions will be temporaries)
      int lastAccessID = versions->back();
      iir_->getMetaData()->getTemporaryFieldAccessIDSet().insert(lastAccessID);
      iir_->getMetaData()->getAllocatedFieldAccessIDSet().erase(lastAccessID);

      // The field with version 0 contains the original name
      const std::string& originalName = iir_->getMetaData()->getNameFromAccessID(versions->front());

      // Register the new field
      iir_->getMetaData()->setAccessIDNamePairOfField(
          newAccessID, originalName + "_" + std::to_string(versions->size()), false);
      iir_->getMetaData()->getAllocatedFieldAccessIDSet().insert(newAccessID);

      versions->push_back(newAccessID);
      iir_->getMetaData()->getVariableVersions().insert(newAccessID, versions);

    } else {
      const std::string& originalName = iir_->getMetaData()->getNameFromAccessID(AccessID);

      // Register the new *and* old field as being multi-versioned and indicate code-gen it has to
      // allocate the second version
      auto versionsVecPtr = std::make_shared<std::vector<int>>();
      *versionsVecPtr = {AccessID, newAccessID};

      iir_->getMetaData()->setAccessIDNamePairOfField(newAccessID, originalName + "_1", false);
      iir_->getMetaData()->getAllocatedFieldAccessIDSet().insert(newAccessID);

      iir_->getMetaData()->getVariableVersions().insert(AccessID, versionsVecPtr);
      iir_->getMetaData()->getVariableVersions().insert(newAccessID, versionsVecPtr);
    }
  } else {
    if(iir_->getMetaData()->getVariableVersions().hasVariableMultipleVersions(AccessID)) {
      // Variable is already multi-versioned, append a new version
      auto versions = iir_->getMetaData()->getVariableVersions().getVersions(AccessID);

      // The variable with version 0 contains the original name
      const std::string& originalName = iir_->getMetaData()->getNameFromAccessID(versions->front());

      // Register the new variable
      iir_->getMetaData()->setAccessIDNamePair(newAccessID, originalName + "_" +
                                                                std::to_string(versions->size()));
      versions->push_back(newAccessID);
      iir_->getMetaData()->getVariableVersions().insert(newAccessID, versions);

    } else {
      const std::string& originalName = iir_->getMetaData()->getNameFromAccessID(AccessID);

      // Register the new *and* old variable as being multi-versioned
      auto versionsVecPtr = std::make_shared<std::vector<int>>();
      *versionsVecPtr = {AccessID, newAccessID};

      iir_->getMetaData()->setAccessIDNamePair(newAccessID, originalName + "_1");
      iir_->getMetaData()->getVariableVersions().insert(AccessID, versionsVecPtr);
      iir_->getMetaData()->getVariableVersions().insert(newAccessID, versionsVecPtr);
    }
  }

  // Rename the Expression
  renameAccessIDInExpr(iir_, AccessID, newAccessID, expr);

  // Recompute the accesses of the current statement (only works with single Do-Methods - for now)
  computeAccesses(iir_,
                  stencil->getStage(curStageIdx)->getSingleDoMethod().getChildren()[curStmtIdx]);

  // Rename the statement and accesses
  for(int stageIdx = curStageIdx;
      dir == RD_Above ? (stageIdx >= 0) : (stageIdx < stencil->getNumStages());
      dir == RD_Above ? stageIdx-- : stageIdx++) {
    iir::Stage& stage = *stencil->getStage(stageIdx);
    iir::DoMethod& doMethod = stage.getSingleDoMethod();

    if(stageIdx == curStageIdx) {
      for(int i = dir == RD_Above ? (curStmtIdx - 1) : (curStmtIdx + 1);
          dir == RD_Above ? (i >= 0) : (i < doMethod.getChildren().size());
          dir == RD_Above ? (--i) : (++i)) {
        renameAccessIDInStmts(iir_, AccessID, newAccessID, doMethod.getChildren()[i]);
        renameAccessIDInAccesses(AccessID, newAccessID, doMethod.getChildren()[i]);
      }

    } else {
      renameAccessIDInStmts(iir_, AccessID, newAccessID, doMethod.getChildren());
      renameAccessIDInAccesses(AccessID, newAccessID, doMethod.getChildren());
    }

    // Updat the fields of the stage
    stage.update(iir::NodeUpdateType::level);
  }

  return newAccessID;
}


} // namespace dawn
