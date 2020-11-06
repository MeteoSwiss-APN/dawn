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
#include "dawn/IIR/AST.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/IIR/AccessComputation.h"
#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/IIR/Extents.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/CreateVersionAndRename.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Exception.h"
#include "dawn/Support/Logger.h"

#include <set>
#include <sstream>

namespace dawn {

namespace {

/// @brief Register all referenced AccessIDs
struct AccessIDGetter : public iir::ASTVisitorForwarding {
  const iir::StencilMetaInformation& metadata_;
  std::set<int> AccessIDs;

  AccessIDGetter(const iir::StencilMetaInformation& metadata) : metadata_(metadata) {}

  virtual void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) override {
    AccessIDs.insert(iir::getAccessID(expr));
  }
};

/// @brief Compute the AccessIDs of the left and right hand side expression of the assignment
static void getAccessIDFromAssignment(const iir::StencilMetaInformation& metadata,
                                      const std::shared_ptr<iir::AssignmentExpr>& assignment,
                                      std::set<int>& LHSAccessIDs, std::set<int>& RHSAccessIDs) {
  auto computeAccessIDs = [&](const std::shared_ptr<iir::Expr>& expr, std::set<int>& AccessIDs) {
    AccessIDGetter getter{metadata};
    expr->accept(getter);
    AccessIDs = std::move(getter.AccessIDs);
  };

  computeAccessIDs(assignment->getLeft(), LHSAccessIDs);
  computeAccessIDs(assignment->getRight(), RHSAccessIDs);
}

/// @brief Compute the AccessIDs of the left and right hand side expression of the assignment
static void getAccessIDFromAssignment(const iir::StencilMetaInformation& metadata,
                                      const std::shared_ptr<iir::VarDeclStmt>& varDecl,
                                      std::set<int>& LHSAccessIDs, std::set<int>& RHSAccessIDs) {
  auto computeAccessIDs = [&](const std::shared_ptr<iir::Expr>& expr, std::set<int>& AccessIDs) {
    AccessIDGetter getter{metadata};
    expr->accept(getter);
    AccessIDs = std::move(getter.AccessIDs);
  };

  LHSAccessIDs.insert(iir::getAccessID(varDecl));
  for(const auto& expr : varDecl->getInitList()) {
    computeAccessIDs(expr, RHSAccessIDs);
  }
}

/// @brief Check if the extent is a stencil-extent (i.e non-pointwise in the horizontal and a
/// counter loop access in the vertical, or vertically indirected)
static bool isHorizontalStencilOrCounterLoopOrderExtent(const iir::Extents& extent,
                                                        iir::LoopOrderKind loopOrder) {
  return extent.verticalExtent().isUndefined() || !extent.isHorizontalPointwise() ||
         extent.getVerticalLoopOrderAccesses(loopOrder).CounterLoopOrder;
}

/// @brief Check if the extent is a stencil-extent (i.e non-pointwise in the horizontal and a
/// counter loop access in the vertical, or vertically indirected)
static bool isHorizontalStencil(const iir::Extents& extent) {
  return !extent.isHorizontalPointwise();
}

/// @brief Report a race condition in the given `statement`
static void reportRaceCondition(const std::shared_ptr<iir::Stmt>& statement,
                                iir::StencilInstantiation& instantiation) {
  std::stringstream ss;
  if(isa<iir::IfStmt>(statement.get())) {
    ss << "Unresolvable race-condition in body of if-statement\n";
  } else {
    ss << "Unresolvable race-condition in statement\n";
  }

  // Print stack trace of stencil calls
  dawn::DiagnosticStack stack;
  if(statement->getData<iir::IIRStmtData>().StackTrace) {
    const auto& stackTrace = *statement->getData<iir::IIRStmtData>().StackTrace;
    for(const auto& frame : stackTrace)
      stack.emplace(std::make_tuple(frame->Callee, frame->Loc));
  }

  throw SemanticError(ss.str() + createDiagnosticStackTrace(
                                     "detected during instantiation of stencil call: ", stack),
                      instantiation.getMetaData().getFileName(), statement->getSourceLocation());
}

} // namespace

void insertGraphStmt(iir::DependencyGraphAccesses& graph, const std::shared_ptr<iir::Stmt>& stmt,
                     iir::LoopOrderKind loopOrder) {

  if(!stmt->getChildren().empty()) {
    for(const auto& s : stmt->getChildren())
      insertGraphStmt(graph, s, loopOrder);
  } else {
    const auto& callerAccesses = stmt->getData<iir::IIRStmtData>().CallerAccesses;

    for(const auto& writeAccess : callerAccesses->getWriteAccesses()) {
      graph.insertNode(writeAccess.first);

      for(const auto& readAccess : callerAccesses->getReadAccesses())
        if(!readAccess.second.getVerticalLoopOrderAccesses(loopOrder).LoopOrder)
          graph.insertEdge(writeAccess.first, readAccess.first, readAccess.second);
    }
  }
}

bool PassFieldVersioning::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const Options& options) {
  numRenames_ = 0;

  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
    iir::Stencil& stencil = *stencilPtr;

    // Iterate multi-stages backwards
    int stageIdx = stencil.getNumStages() - 1;
    for(auto multiStageRit = stencil.childrenRBegin(), multiStageRend = stencil.childrenREnd();
        multiStageRit != multiStageRend; ++multiStageRit) {
      iir::MultiStage& multiStage = (**multiStageRit);
      iir::LoopOrderKind loopOrder = multiStage.getLoopOrder();

      iir::DependencyGraphAccesses newGraph(stencilInstantiation->getMetaData());
      auto oldGraph = newGraph;

      // Iterate stages bottom -> top
      for(auto stageRit = multiStage.childrenRBegin(), stageRend = multiStage.childrenREnd();
          stageRit != stageRend; ++stageRit) {
        iir::Stage& stage = (**stageRit);
        iir::DoMethod& doMethod = stage.getSingleDoMethod();

        // Iterate statements bottom -> top
        for(int stmtIndex = doMethod.getAST().getStatements().size() - 1; stmtIndex >= 0;
            --stmtIndex) {
          oldGraph = newGraph;

          const auto& stmt = doMethod.getAST().getStatements()[stmtIndex];
          insertGraphStmt(newGraph, stmt, loopOrder);

          // Try to resolve race-conditions by using double buffering if necessary
          auto rc = fixRaceCondition(stencilInstantiation, newGraph, stencil, doMethod, loopOrder,
                                     stageIdx, stmtIndex, options.DumpRaceConditionGraph);

          if(rc == RCKind::Unresolvable) {
            // Nothing we can do ... bail out
            return false;
          } else if(rc == RCKind::Fixed) {
            // We fixed a race condition (this means some fields have changed and our current graph
            // is invalid)
            newGraph = oldGraph;
            insertGraphStmt(newGraph, stmt, loopOrder);
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

  if(numRenames_ == 0)
    DAWN_LOG(INFO) << stencilInstantiation->getName() << ": no rename";

  return true;
}

PassFieldVersioning::RCKind PassFieldVersioning::fixRaceCondition(
    const std::shared_ptr<iir::StencilInstantiation> instantiation,
    iir::DependencyGraphAccesses const& graph, iir::Stencil& stencil, iir::DoMethod& doMethod,
    iir::LoopOrderKind loopOrder, int stageIdx, int index, bool dump) {
  using Vertex = iir::DependencyGraphAccesses::Vertex;
  using Edge = iir::DependencyGraphAccesses::Edge;

  const std::shared_ptr<iir::Stmt>& statement = doMethod.getAST().getStatements()[index];

  int numRenames = 0;

  // Find all strongly connected components in the graph ...
  auto SCCs = std::make_unique<std::vector<std::set<int>>>();
  graph.findStronglyConnectedComponents(*SCCs);

  // Check if we have self dependencies e.g `u = u(i+1)` (Our SCC algorithm does not capture SCCs
  // of size one i.e single nodes)
  for(const auto& AccessIDVertexPair : graph.getVertices()) {
    const Vertex& vertex = AccessIDVertexPair.second;

    for(const Edge& edge : graph.getAdjacencyList()[vertex.VertexID]) {
      if(edge.FromVertexID == edge.ToVertexID && isHorizontalStencil(edge.Data)) {
        SCCs->emplace_back(std::set<int>{vertex.Value});
        break;
      }
    }
  }
 
  if(SCCs->empty())
    return RCKind::Nothing;

  // Check whether our statement is an `ExprStmt` and contains an `AssignmentExpr`. If not,
  // we cannot perform any double buffering (e.g if there is a problem inside an `IfStmt`, nothing
  // we can do (yet ;))
  std::shared_ptr<iir::AssignmentExpr> assignment;
  std::shared_ptr<iir::VarDeclStmt> varDecl;

  std::set<int> LHSAccessIDs, RHSAccessIDs;

  if(iir::ExprStmt* stmt = dyn_cast<iir::ExprStmt>(statement.get())) {
    assignment = std::dynamic_pointer_cast<iir::AssignmentExpr>(stmt->getExpr());
    // Get AccessIDs of the LHS and RHS
    getAccessIDFromAssignment(instantiation->getMetaData(), assignment, LHSAccessIDs, RHSAccessIDs);

  } else if(varDecl = std::dynamic_pointer_cast<iir::VarDeclStmt>(statement)) {
    // Get AccessIDs of the LHS and RHS
    getAccessIDFromAssignment(instantiation->getMetaData(), varDecl, LHSAccessIDs, RHSAccessIDs);
  }

  if(!assignment && !varDecl) {
    if(dump)
      graph.toDot("rc_" + instantiation->getName() + ".dot");
    reportRaceCondition(statement, *instantiation);
    // The function call above throws, so do not need a return here any longer. Will refactor
    // further later. return RCKind::Unresolvable;
  }

  DAWN_ASSERT_MSG(LHSAccessIDs.size() == 1, "left hand side should only have only one AccessID");
  int LHSAccessID = *LHSAccessIDs.begin();

  DAWN_ASSERT_MSG(SCCs->size() == 1, "only one strongly connected component can be handled");
  std::set<int>& SCC = (*SCCs)[0];

  // If the LHSAccessID is not part of the SCC, we cannot resolve the race-condition
  if(!SCC.count(LHSAccessID)) {
    if(dump)
      graph.toDot("rc_" + instantiation->getName() + ".dot");
    reportRaceCondition(statement, *instantiation);
    // The function call above throws, so do not need a return here any longer. Will refactor
    // further later. return RCKind::Unresolvable;
  }

  std::set<int> renameCandiates;
  for(int AccessID : SCC) {
    if(RHSAccessIDs.count(AccessID))
      renameCandiates.insert(AccessID);
  }

  std::stringstream ss;
  ss << instantiation->getName() << ": rename:";
  std::vector<std::shared_ptr<iir::Expr>> replExprs;
  if(assignment) {
    replExprs.push_back(assignment->getRight());
  } else if(varDecl) {
    for(auto& expr : varDecl->getInitList()) {
      replExprs.push_back(expr);
    }
  }

  // Create a new multi-versioned field and rename all occurences
  for(int oldAccessID : renameCandiates) {
    for(auto& expr : replExprs) {
      int newAccessID = createVersionAndRename(instantiation.get(), oldAccessID, &stencil, stageIdx,
                                               index, expr, RenameDirection::Above);

      ss << (numRenames != 0 ? ", " : " ")
         << instantiation->getMetaData().getFieldNameFromAccessID(oldAccessID) << ":"
         << instantiation->getMetaData().getFieldNameFromAccessID(newAccessID);

      numRenames++;
    }
  }

  if(numRenames > 0)
    DAWN_DIAG(INFO, instantiation->getMetaData().getFileName(), statement->getSourceLocation())
        << ss.str();
  else
    DAWN_DIAG(INFO, instantiation->getMetaData().getFileName(), statement->getSourceLocation())
        << instantiation->getName() << ": No renames performed";

  numRenames_ += numRenames;
  return RCKind::Fixed;
}

} // namespace dawn
