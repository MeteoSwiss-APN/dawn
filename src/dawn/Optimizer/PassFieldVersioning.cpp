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
#include "dawn/Optimizer/AccessComputation.h"
#include "dawn/Optimizer/DependencyGraphAccesses.h"
#include "dawn/Optimizer/Extents.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/StatementAccessesPair.h"
#include "dawn/Optimizer/Stencil.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/SIR/AST.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include <iostream>
#include <set>

namespace dawn {

namespace {

/// @brief Register all referenced AccessIDs
struct AccessIDGetter : public ASTVisitorForwarding {
  const StencilInstantiation& Instantiation;
  std::set<int> AccessIDs;

  AccessIDGetter(const StencilInstantiation& instantiation) : Instantiation(instantiation) {}

  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) override {
    AccessIDs.insert(Instantiation.getAccessIDFromExpr(expr));
  }
};

/// @brief Compute the AccessIDs of the left and right hand side expression of the assignment
static void getAccessIDFromAssignment(const StencilInstantiation& instantiation,
                                      AssignmentExpr* assignment, std::set<int>& LHSAccessIDs,
                                      std::set<int>& RHSAccessIDs) {
  auto computeAccessIDs = [&](const std::shared_ptr<Expr>& expr, std::set<int>& AccessIDs) {
    AccessIDGetter getter{instantiation};
    expr->accept(getter);
    AccessIDs = std::move(getter.AccessIDs);
  };

  computeAccessIDs(assignment->getLeft(), LHSAccessIDs);
  computeAccessIDs(assignment->getRight(), RHSAccessIDs);
}

/// @brief Check if the extent is a stencil-extent (i.e non-pointwise in the horizontal and a
/// counter loop acesses in the vertical)
static bool isHorizontalStencilOrCounterLoopOrderExtent(const Extents& extent,
                                                        LoopOrderKind loopOrder) {
  return !extent.isHorizontalPointwise() ||
         extent.getVerticalLoopOrderAccesses(loopOrder).CounterLoopOrder;
}

/// @brief Report a race condition in the given `statement`
static void reportRaceCondition(const Statement& statement, StencilInstantiation& instantiation) {
  DiagnosticsBuilder diag(DiagnosticsKind::Error, statement.ASTStmt->getSourceLocation());

  if(isa<IfStmt>(statement.ASTStmt.get())) {
    diag << "unresolvable race-condition in body of if-statement";
  } else {
    diag << "unresolvable race-condition in statement";
  }

  instantiation.getOptimizerContext()->getDiagnostics().report(diag);

  // Print stack trace of stencil calls
  if(statement.StackTrace) {
    std::vector<sir::StencilCall*>& stackTrace = *statement.StackTrace;
    for(int i = stackTrace.size() - 1; i >= 0; --i) {
      DiagnosticsBuilder note(DiagnosticsKind::Note, stackTrace[i]->Loc);
      note << "detected during instantiation of stencil-call '" << stackTrace[i]->Callee << "'";
      instantiation.getOptimizerContext()->getDiagnostics().report(note);
    }
  }
}

} // anonymous namespace

PassFieldVersioning::PassFieldVersioning() : Pass("PassFieldVersioning", true), numRenames_(0) {}

bool PassFieldVersioning::run(const std::shared_ptr<StencilInstantiation>& stencilInstantiation) {
  OptimizerContext* context = stencilInstantiation->getOptimizerContext();
  numRenames_ = 0;

  for(auto& stencilPtr : stencilInstantiation->getStencils()) {
    Stencil& stencil = *stencilPtr;

    // Iterate multi-stages backwards
    int stageIdx = stencil.getNumStages() - 1;
    for(auto multiStageRit = stencil.getMultiStages().rbegin(),
             multiStageRend = stencil.getMultiStages().rend();
        multiStageRit != multiStageRend; ++multiStageRit) {
      MultiStage& multiStage = (**multiStageRit);
      LoopOrderKind loopOrder = multiStage.getLoopOrder();

      std::shared_ptr<DependencyGraphAccesses> newGraph, oldGraph;
      newGraph = std::make_shared<DependencyGraphAccesses>(stencilInstantiation.get());

      // Iterate stages bottom -> top
      for(auto stageRit = multiStage.getStages().rbegin(),
               stageRend = multiStage.getStages().rend();
          stageRit != stageRend; ++stageRit) {
        Stage& stage = (**stageRit);
        DoMethod& doMethod = stage.getSingleDoMethod();

        // Iterate statements bottom -> top
        for(int stmtIndex = doMethod.getStatementAccessesPairs().size() - 1; stmtIndex >= 0;
            --stmtIndex) {
          oldGraph = newGraph->clone();

          auto& stmtAccessesPair = doMethod.getStatementAccessesPairs()[stmtIndex];
          newGraph->insertStatementAccessesPair(stmtAccessesPair);

          // Try to resolve race-conditions by using double buffering if necessary
          auto rc =
              fixRaceCondition(newGraph.get(), stencil, doMethod, loopOrder, stageIdx, stmtIndex);

          if(rc == RCKind::RK_Unresolvable)
            // Nothing we can do ... bail out
            return false;
          else if(rc == RCKind::RK_Fixed) {
            // We fixed a race condition (this means some fields have changed and our current graph
            // is invalid)
            newGraph = oldGraph;
            newGraph->insertStatementAccessesPair(stmtAccessesPair);
          }
        }
      }
      stageIdx--;
    }
  }

  if(context->getOptions().ReportPassFieldVersioning && numRenames_ == 0)
    std::cout << "\nPASS: " << getName() << ": " << stencilInstantiation->getName()
              << ": no rename\n";
  return true;
}

PassFieldVersioning::RCKind
PassFieldVersioning::fixRaceCondition(const DependencyGraphAccesses* graph, Stencil& stencil,
                                      DoMethod& doMethod, LoopOrderKind loopOrder, int stageIdx,
                                      int index) {
  using Vertex = DependencyGraphAccesses::Vertex;
  using Edge = DependencyGraphAccesses::Edge;

  Statement& statement = *doMethod.getStatementAccessesPairs()[index]->getStatement();

  auto& instantiation = stencil.getStencilInstantiation();
  OptimizerContext* context = instantiation.getOptimizerContext();
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
    if(context->getOptions().DumpRaceConditionGraph)
      graph->toDot("rc_" + instantiation.getName() + ".dot");
    reportRaceCondition(statement, instantiation);
    return RCKind::RK_Unresolvable;
  }

  // Get AccessIDs of the LHS and RHS
  std::set<int> LHSAccessIDs, RHSAccessIDs;
  getAccessIDFromAssignment(instantiation, assignment, LHSAccessIDs, RHSAccessIDs);

  DAWN_ASSERT_MSG(LHSAccessIDs.size() == 1, "left hand side should only have only one AccessID");
  int LHSAccessID = *LHSAccessIDs.begin();

  // If the LHSAccessID is not part of the SCC, we cannot resolve the race-condition
  for(std::set<int>& scc : *stencilSCCs) {
    if(!scc.count(LHSAccessID)) {
      if(context->getOptions().DumpRaceConditionGraph)
        graph->toDot("rc_" + instantiation.getName() + ".dot");
      reportRaceCondition(statement, instantiation);
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

  if(context->getOptions().ReportPassFieldVersioning)
    std::cout << "\nPASS: " << getName() << ": " << instantiation.getName()
              << ": rename:" << statement.ASTStmt->getSourceLocation().Line;

  // Create a new multi-versioned field and rename all occurences
  for(int oldAccessID : renameCandiates) {
    int newAccessID = instantiation.createVersionAndRename(oldAccessID, &stencil, stageIdx, index,
                                                           assignment->getRight(),
                                                           StencilInstantiation::RD_Above);

    if(context->getOptions().ReportPassFieldVersioning)
      std::cout << (numRenames != 0 ? ", " : " ") << instantiation.getNameFromAccessID(oldAccessID)
                << ":" << instantiation.getNameFromAccessID(newAccessID);

    numRenames++;
  }

  if(context->getOptions().ReportPassFieldVersioning && numRenames > 0)
    std::cout << "\n";

  numRenames_ += numRenames;
  return RCKind::RK_Fixed;
}

} // namespace dawn
