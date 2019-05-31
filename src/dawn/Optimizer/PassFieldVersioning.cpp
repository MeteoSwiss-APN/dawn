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
  const iir::StencilMetaInformation& metadata_;
  std::set<int> AccessIDs;

  AccessIDGetter(const iir::StencilMetaInformation& metadata) : metadata_(metadata) {}

  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) override {
    AccessIDs.insert(metadata_.getAccessIDFromExpr(expr));
  }
};

} // anonymous namespace

/// @brief Compute the AccessIDs of the left and right hand side expression of the assignment
static void getAccessIDFromAssignment(const iir::StencilMetaInformation& metadata,
                                      AssignmentExpr* assignment, std::set<int>& LHSAccessIDs,
                                      std::set<int>& RHSAccessIDs) {
  auto computeAccessIDs = [&](const std::shared_ptr<Expr>& expr, std::set<int>& AccessIDs) {
    AccessIDGetter getter{metadata};
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

static void addAssignmentToDoMethod(std::unique_ptr<iir::DoMethod>& domethod, int assignmentID,
                                    int assigneeID,
                                    const std::shared_ptr<iir::StencilInstantiation>& context,
                                    boost::optional<iir::Extents>& readExtents) {
  // Create the StatementAccessPair of the assignment with the new and old variables
  auto fa_assignee =
      std::make_shared<FieldAccessExpr>(context->getMetaData().getNameFromAccessID(assigneeID));
  auto fa_assignment =
      std::make_shared<FieldAccessExpr>(context->getMetaData().getNameFromAccessID(assignmentID));
  auto assignmentExpression = std::make_shared<AssignmentExpr>(fa_assignment, fa_assignee, "=");
  auto expAssignment = std::make_shared<ExprStmt>(assignmentExpression);
  auto assignmentStatement = std::make_shared<Statement>(expAssignment, nullptr);
  auto pair = make_unique<iir::StatementAccessesPair>(assignmentStatement);
  auto newAccess = std::make_shared<iir::Accesses>();

  // The exetens were computed on a multistage-level and are passed to here. If they are
  // uninitialized, we just copy the compute-domain, otherwise the read-extents of the versioned
  // field determines how much copying needs to happen
  if(readExtents.is_initialized()) {
    newAccess->addWriteExtent(assignmentID, readExtents.get());
    newAccess->addReadExtent(assigneeID, readExtents.get());
  } else {
    newAccess->addWriteExtent(assignmentID, iir::Extents(Array3i{{0, 0, 0}}));
    newAccess->addReadExtent(assigneeID, iir::Extents(Array3i{{0, 0, 0}}));
  }
  pair->setAccesses(newAccess);
  domethod->insertChild(std::move(pair));

  // Add the new expressions to the map
  context->getMetaData().insertExprToAccessID(fa_assignment, assignmentID);
  context->getMetaData().insertExprToAccessID(fa_assignee, assigneeID);
}
/// @brief Creates the stage in which assignment happens (fill and flush)
static std::unique_ptr<iir::Stage>
createAssignmentStage(const iir::Interval& interval, int assignmentID, int assigneeID,
                      const std::shared_ptr<iir::StencilInstantiation>& context,
                      boost::optional<iir::Extents>& readExtents) {
  // Add the stage that assined the assingee to the assignment
  std::unique_ptr<iir::Stage> assignmentStage =
      make_unique<iir::Stage>(context->getMetaData(), context->nextUID(), interval);
  iir::Stage::DoMethodSmartPtr_t domethod =
      make_unique<iir::DoMethod>(interval, context->getMetaData());
  domethod->clearChildren();

  addAssignmentToDoMethod(domethod, assignmentID, assigneeID, context, readExtents);

  // Add the single do method to the new Stage
  assignmentStage->clearChildren();
  assignmentStage->addDoMethod(domethod);
  for(auto& doMethod : assignmentStage->getChildren()) {
    doMethod->update(iir::NodeUpdateType::level);
  }
  assignmentStage->update(iir::NodeUpdateType::level);
  return assignmentStage;
}

PassFieldVersioning::PassFieldVersioning(FieldVersioningPassMode mode)
    : Pass("PassFieldVersioning", true), mode_(mode), numRenames_(0) {}

bool PassFieldVersioning::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  if(mode_ == FieldVersioningPassMode::FM_CreateVersion) {
    OptimizerContext* context = stencilInstantiation->getOptimizerContext();
    numRenames_ = 0;

    for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
      iir::Stencil& stencil = *stencilPtr;

      // Iterate multi-stages backwards
      int stageIdx = stencil.getNumStages() - 1;
      for(auto multiStageRit = stencil.childrenRBegin(), multiStageRend = stencil.childrenREnd();
          multiStageRit != multiStageRend; ++multiStageRit) {
        iir::MultiStage& multiStage = (**multiStageRit);
        iir::LoopOrderKind loopOrder = multiStage.getLoopOrder();

        std::shared_ptr<iir::DependencyGraphAccesses> newGraph, oldGraph;
        newGraph =
            std::make_shared<iir::DependencyGraphAccesses>(stencilInstantiation->getMetaData());

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
            auto rc = fixRaceCondition(stencilInstantiation, newGraph.get(), stencil, doMethod,
                                       loopOrder, stageIdx, stmtIndex);

            if(rc == RCKind::RK_Unresolvable) {
              // Nothing we can do ... bail out
              return false;
            } else if(rc == RCKind::RK_Fixed) {
              // We fixed a race condition (this means some fields have changed and our current
              // graph
              // is invalid)
              newGraph = oldGraph;
              newGraph->insertStatementAccessesPair(stmtAccessesPair);
            }
            doMethod.update(iir::NodeUpdateType::level);
          }
          stage.update(iir::NodeUpdateType::level);
        }
        stageIdx--;
      }

      for(const auto& ms : iterateIIROver<iir::MultiStage>(stencil)) {
        ms->update(iir::NodeUpdateType::levelAndTreeAbove);
      }
    }

    if(context->getOptions().ReportPassFieldVersioning && numRenames_ == 0)
      std::cout << "\nPASS: " << getName() << ": " << stencilInstantiation->getName()
                << ": no rename\n";
    return true;
  } else if(mode_ == FieldVersioningPassMode::FM_FixAccess) {
    for(int id : stencilInstantiation->getMetaData()
                     .getFieldAccessMetadata()
                     .getVariableVersions()
                     .getVersionIDs()) {
      for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
        iir::Stencil& stencil = *stencilPtr;
        for(const auto& mss : iterateIIROver<iir::MultiStage>(stencil)) {
          // For every mulitstage, check if the versioned field as a read-access first
          auto multiInterval = mss->computeReadAccessInterval(id);
          if(multiInterval.empty())
            continue;

          // If it has one, we need to generate a domethod that fills that access from the
          // original field in every interval that is being accessed before it's first write
          auto extents = mss->getField(id).getReadExtentsRB();
          for(auto interval : multiInterval.getIntervals()) {

            // we create the new stage that holds these do-methods
            auto insertedStage = createAssignmentStage(
                interval, id, stencilInstantiation->getMetaData().getOriginalVersionOfAccessID(id),
                stencilInstantiation, extents);
            // and insert them at the beginnning of the MultiStage
            mss->insertChild(mss->childrenBegin(), std::move(insertedStage));
          }
          // update the mss: #TODO: this is still a workaround since we don't have level-and below:
          for(const auto& domethods : iterateIIROver<iir::DoMethod>(*mss)) {
            domethods->update(iir::NodeUpdateType::levelAndTreeAbove);
          }
        }
      }
    }
    return true;
  } else {
    dawn_unreachable("illegal mode for PassFieldVersioning");
    return false;
  }
}

PassFieldVersioning::RCKind PassFieldVersioning::fixRaceCondition(
    const std::shared_ptr<iir::StencilInstantiation> instantiation,
    const iir::DependencyGraphAccesses* graph, iir::Stencil& stencil, iir::DoMethod& doMethod,
    iir::LoopOrderKind loopOrder, int stageIdx, int index) {
  using Vertex = iir::DependencyGraphAccesses::Vertex;
  using Edge = iir::DependencyGraphAccesses::Edge;

  Statement& statement = *doMethod.getChildren()[index]->getStatement();

  OptimizerContext* context = instantiation->getOptimizerContext();
  int numRenames = 0;

  // Vector of strongly connected components with atleast one stencil access
  auto stencilSCCs = make_unique<std::vector<std::set<int>>>();

  // Find all strongly connected components in the graph ...
  auto SCCs = make_unique<std::vector<std::set<int>>>();
  graph->findStronglyConnectedComponents(*SCCs);

  // ... and add those which have atleast one stencil access
  for(std::set<int>& scc : *SCCs) {
    bool isStencilSCC = false;

    for(int fromAccessID : scc) {
      std::size_t fromVertexID = graph->getVertexIDFromValue(fromAccessID);

      for(const Edge& edge : *graph->getAdjacencyList()[fromVertexID]) {
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
    // Check if we have self dependencies e.g `u = u(i+1)` (Our SCC algorithm does not capture
    // SCCs
    // of size one i.e single nodes)
    for(const auto& AccessIDVertexPair : graph->getVertices()) {
      const Vertex& vertex = AccessIDVertexPair.second;

      for(const Edge& edge : *graph->getAdjacencyList()[vertex.VertexID]) {
        if(edge.FromVertexID == edge.ToVertexID &&
           isHorizontalStencilOrCounterLoopOrderExtent(edge.Data, loopOrder)) {
          stencilSCCs->emplace_back(std::set<int>{vertex.value});
          break;
        }
      }
    }
  }

  // If we only have non-stencil SCCs and there are no input and output fields (i.e we don't
  // have a
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
  // we cannot perform any double buffering (e.g if there is a problem inside an `IfStmt`,
  // nothing
  // we can do (yet ;))
  AssignmentExpr* assignment = nullptr;
  if(ExprStmt* stmt = dyn_cast<ExprStmt>(statement.ASTStmt.get()))
    assignment = dyn_cast<AssignmentExpr>(stmt->getExpr().get());

  if(!assignment) {
    if(context->getOptions().DumpRaceConditionGraph)
      graph->toDot("rc_" + instantiation->getName() + ".dot");
    reportRaceCondition(statement, *instantiation);
    return RCKind::RK_Unresolvable;
  }

  // Get AccessIDs of the LHS and RHS
  std::set<int> LHSAccessIDs, RHSAccessIDs;
  getAccessIDFromAssignment(instantiation->getMetaData(), assignment, LHSAccessIDs, RHSAccessIDs);

  DAWN_ASSERT_MSG(LHSAccessIDs.size() == 1, "left hand side should only have only one AccessID");
  int LHSAccessID = *LHSAccessIDs.begin();

  // If the LHSAccessID is not part of the SCC, we cannot resolve the race-condition
  for(std::set<int>& scc : *stencilSCCs) {
    if(!scc.count(LHSAccessID)) {
      if(context->getOptions().DumpRaceConditionGraph)
        graph->toDot("rc_" + instantiation->getName() + ".dot");
      reportRaceCondition(statement, *instantiation);
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
    std::cout << "\nPASS: " << getName() << ": " << instantiation->getName()
              << ": rename:" << statement.ASTStmt->getSourceLocation().Line;

  // Create a new multi-versioned field and rename all occurences
  for(int oldAccessID : renameCandiates) {
    int newAccessID = instantiation->createVersionAndRename(oldAccessID, &stencil, stageIdx, index,
                                                            assignment->getRight(),
                                                            iir::StencilInstantiation::RD_Above);

    if(context->getOptions().ReportPassFieldVersioning)
      std::cout << (numRenames != 0 ? ", " : " ")
                << instantiation->getMetaData().getFieldNameFromAccessID(oldAccessID) << ":"
                << instantiation->getMetaData().getFieldNameFromAccessID(newAccessID);

    numRenames++;
  }

  if(context->getOptions().ReportPassFieldVersioning && numRenames > 0)
    std::cout << "\n";

  numRenames_ += numRenames;
  return RCKind::RK_Fixed;
}

} // namespace dawn
