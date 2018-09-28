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
#include "dawn/IIR/Extents.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/AccessComputation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/SIR/AST.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include <iostream>
#include <set>

namespace dawn {

namespace {

/// @brief Register all referenced AccessIDs
struct AccessIDGetter : public ASTVisitorForwarding {
  const iir::StencilInstantiation& Instantiation;
  std::set<int> AccessIDs;

  AccessIDGetter(const iir::StencilInstantiation& instantiation) : Instantiation(instantiation) {}

  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) override {
    AccessIDs.insert(Instantiation.getAccessIDFromExpr(expr));
  }
};

/// @brief Compute the AccessIDs of the left and right hand side expression of the assignment
static void getAccessIDFromAssignment(const iir::StencilInstantiation& instantiation,
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
static bool isHorizontalStencilOrCounterLoopOrderExtent(const iir::Extents& extent,
                                                        iir::LoopOrderKind loopOrder) {
  return !extent.isHorizontalPointwise() ||
         extent.getVerticalLoopOrderAccesses(loopOrder).CounterLoopOrder;
}

/// @brief Report a race condition in the given `statement`
static void reportRaceCondition(const Statement& statement,
                                iir::StencilInstantiation& instantiation) {
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

bool PassFieldVersioning::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  instantiation_ = stencilInstantiation.get();
  OptimizerContext* context = stencilInstantiation->getOptimizerContext();
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
      newGraph = std::make_shared<iir::DependencyGraphAccesses>(stencilInstantiation.get());

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
          auto rc =
              fixRaceCondition(newGraph.get(), stencil, doMethod, loopOrder, stageIdx, stmtIndex);

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

  // Now we need to check that every field is synchronized: if the first access to the versioned
  // field is a write-access, we don't need to do anything. If it is a read-access, we need to
  // insert a stage before that access where we move the data to the field.
  if(!versionedIDoriginalIDs_.empty()) {
    // check first access:
    for(auto IDPair : versionedIDoriginalIDs_) {
      bool IDhandeled = false;
      for(auto& stencil : instantiation_->getIIR()->getChildren()) {
        for(auto mssptr = stencil->childrenBegin(); mssptr != stencil->childrenEnd(); ++mssptr) {
          std::unique_ptr<iir::MultiStage>& mss = *mssptr;
          auto retval = checkReadBeforeWrite(IDPair.first, mss);
          if(retval) {
            // insert stage
            auto newmss =
                createAssignmentMS(IDPair.first, IDPair.second, mss->getEnclosingInterval());
            stencil->insertChild(mssptr, std::move(newmss));
            // This field is handled, we move on to the next one
            IDhandeled = true;
            break;
          }
          //          else if(retval == FieldVersionFirstAccessKind::FA_Write) {
          //            //  We have a write first, no need to do anything, we move on
          //            IDhandeled = true;
          //            break;
          //          }
        }
        if(IDhandeled) {
          break;
        }
      }
    }
  }

  if(context->getOptions().ReportPassFieldVersioning && numRenames_ == 0)
    std::cout << "\nPASS: " << getName() << ": " << stencilInstantiation->getName()
              << ": no rename\n";
  return true;
}

PassFieldVersioning::RCKind
PassFieldVersioning::fixRaceCondition(const iir::DependencyGraphAccesses* graph,
                                      iir::Stencil& stencil, iir::DoMethod& doMethod,
                                      iir::LoopOrderKind loopOrder, int stageIdx, int index) {
  using Vertex = iir::DependencyGraphAccesses::Vertex;
  using Edge = iir::DependencyGraphAccesses::Edge;

  Statement& statement = *doMethod.getChildren()[index]->getStatement();

  //  auto& instantiation = stencil.getStencilInstantiation();
  OptimizerContext* context = instantiation_->getOptimizerContext();
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
      graph->toDot("rc_" + instantiation_->getName() + ".dot");
    reportRaceCondition(statement, *instantiation_);
    return RCKind::RK_Unresolvable;
  }

  // Get AccessIDs of the LHS and RHS
  std::set<int> LHSAccessIDs, RHSAccessIDs;
  getAccessIDFromAssignment(*instantiation_, assignment, LHSAccessIDs, RHSAccessIDs);

  DAWN_ASSERT_MSG(LHSAccessIDs.size() == 1, "left hand side should only have only one AccessID");
  int LHSAccessID = *LHSAccessIDs.begin();

  // If the LHSAccessID is not part of the SCC, we cannot resolve the race-condition
  for(std::set<int>& scc : *stencilSCCs) {
    if(!scc.count(LHSAccessID)) {
      if(context->getOptions().DumpRaceConditionGraph)
        graph->toDot("rc_" + instantiation_->getName() + ".dot");
      reportRaceCondition(statement, *instantiation_);
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
    std::cout << "\nPASS: " << getName() << ": " << instantiation_->getName()
              << ": rename:" << statement.ASTStmt->getSourceLocation().Line;

  // Create a new multi-versioned field and rename all occurences
  for(int oldAccessID : renameCandiates) {
    int newAccessID = instantiation_->createVersionAndRename(oldAccessID, &stencil, stageIdx, index,
                                                             assignment->getRight(),
                                                             iir::StencilInstantiation::RD_Above);

    if(context->getOptions().ReportPassFieldVersioning)
      std::cout << (numRenames != 0 ? ", " : " ")
                << instantiation_->getNameFromAccessID(oldAccessID) << ":"
                << instantiation_->getNameFromAccessID(newAccessID);

    numRenames++;
  }

  if(context->getOptions().ReportPassFieldVersioning && numRenames > 0)
    std::cout << "\n";

  numRenames_ += numRenames;
  return RCKind::RK_Fixed;
}

bool PassFieldVersioning::checkReadBeforeWrite(int AccessID,
                                               const std::unique_ptr<iir::MultiStage>& mss) {

  for(const auto& stmtAccessesPair : iterateIIROver<iir::StatementAccessesPair>(*mss)) {

    // Find first if this statement has a read
    auto readAccessIterator =
        stmtAccessesPair->getCallerAccesses()->getReadAccesses().find(AccessID);
    if(readAccessIterator != stmtAccessesPair->getCallerAccesses()->getReadAccesses().end()) {
      return true;
    }
    // If we did not find a read statement so far, we have  a write first and do not need to
    // fill the cache
    auto wirteAccessIterator =
        stmtAccessesPair->getCallerAccesses()->getWriteAccesses().find(AccessID);
    if(wirteAccessIterator != stmtAccessesPair->getCallerAccesses()->getWriteAccesses().end()) {
      return false;
    }
  }
  return false;
}

/// @brief Creates the stage in which assignment happens (fill and flush)
std::unique_ptr<iir::Stage>
PassFieldVersioning::createAssignmentStage(const iir::Interval& interval,
                                           const std::vector<int>& assignmentIDs,
                                           const std::vector<int>& assigneeIDs) {
  // Add the cache Flush stage
  std::unique_ptr<iir::Stage> assignmentStage =
      make_unique<iir::Stage>(*instantiation_, instantiation_->nextUID(), interval);
  iir::Stage::DoMethodSmartPtr_t domethod = make_unique<iir::DoMethod>(interval);
  domethod->clearChildren();

  for(int i = 0; i < assignmentIDs.size(); ++i) {
    int assignmentID = assignmentIDs[i];
    int assigneeID = assigneeIDs[i];
    addAssignmentToDoMethod(domethod, assignmentID, assigneeID);
  }

  // Add the single do method to the new Stage
  assignmentStage->clearChildren();
  assignmentStage->addDoMethod(domethod);
  assignmentStage->update(iir::NodeUpdateType::level);

  return assignmentStage;
}

///@brief Add the assignment operator of two unique id's to a given domethod
void PassFieldVersioning::addAssignmentToDoMethod(const iir::Stage::DoMethodSmartPtr_t& domethod,
                                                  int assignmentID, int assigneeID) {
  // Create the StatementAccessPair of the assignment with the new and old variables
  auto fa_assignee =
      std::make_shared<FieldAccessExpr>(instantiation_->getNameFromAccessID(assigneeID));
  auto fa_assignment =
      std::make_shared<FieldAccessExpr>(instantiation_->getNameFromAccessID(assignmentID));
  auto assignmentExpression = std::make_shared<AssignmentExpr>(fa_assignment, fa_assignee, "=");
  auto expAssignment = std::make_shared<ExprStmt>(assignmentExpression);
  auto assignmentStatement = std::make_shared<Statement>(expAssignment, nullptr);
  auto pair = make_unique<iir::StatementAccessesPair>(assignmentStatement);
  auto newAccess = std::make_shared<iir::Accesses>();
  newAccess->addWriteExtent(assignmentID, iir::Extents(Array3i{{0, 0, 0}}));
  newAccess->addReadExtent(assigneeID, iir::Extents(Array3i{{0, 0, 0}}));
  pair->setAccesses(newAccess);
  domethod->insertChild(std::move(pair));

  // Add the new expressions to the map
  instantiation_->mapExprToAccessID(fa_assignment, assignmentID);
  instantiation_->mapExprToAccessID(fa_assignee, assigneeID);
}

std::unique_ptr<iir::MultiStage>
PassFieldVersioning::createAssignmentMS(int assignmentID, int assigneeID, iir::Interval interval) {
  std::unique_ptr<iir::MultiStage> assignmentMSS =
      make_unique<iir::MultiStage>(*instantiation_, iir::LoopOrderKind::LK_Parallel);
  // change the Enclosing Interval to ECI of the caller!
  std::unique_ptr<iir::Stage> cacheFillStage =
      createAssignmentStage(interval, {assignmentID}, {assigneeID});
  auto stageBegin = assignmentMSS->childrenBegin();
  assignmentMSS->insertChild(stageBegin, std::move(cacheFillStage));
  return assignmentMSS;
}

} // namespace dawn
