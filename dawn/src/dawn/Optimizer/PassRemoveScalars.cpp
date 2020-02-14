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

#include "PassRemoveScalars.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTStmt.h"
#include "dawn/IIR/AccessComputation.h"
#include "dawn/IIR/DoMethod.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/IIR/StencilMetaInformation.h"

#include <tuple>
#include <unordered_map>

namespace dawn {
namespace {

class VarAccessesReplacer : public ast::ASTVisitorPostOrder {
  const int variableToSubstitute_;
  const std::shared_ptr<const iir::Expr> substituteExpr_;

public:
  std::shared_ptr<iir::Expr>
  postVisitNode(std::shared_ptr<iir::VarAccessExpr> const& expr) override {
    if(iir::getAccessID(expr) == variableToSubstitute_) {
      return substituteExpr_->clone();
    }
    return expr;
  }

  VarAccessesReplacer(int variableToSubstitute,
                      const std::shared_ptr<const iir::Expr> substituteExpr)
      : variableToSubstitute_(variableToSubstitute), substituteExpr_(substituteExpr) {}
};

// Returns the access ID of the scalar variable that `stmt` writes to.
// Returns std::nullopt if `stmt` is not writing to a scalar variable.
std::optional<int> getScalarWrittenByStatement(const std::shared_ptr<iir::Stmt> stmt,
                                               const iir::StencilMetaInformation& metadata) {
  const auto& idToLocalVariableData = metadata.getAccessIDToLocalVariableDataMap();
  std::optional<int> scalarAccessID;

  if(stmt->getKind() == iir::Stmt::Kind::VarDeclStmt) {
    scalarAccessID = iir::getAccessID(std::dynamic_pointer_cast<iir::VarDeclStmt>(stmt));
  } else if(stmt->getKind() == iir::Stmt::Kind::ExprStmt) {
    const auto& exprStmt = std::dynamic_pointer_cast<iir::ExprStmt>(stmt);
    if(exprStmt->getExpr()->getKind() == iir::Expr::Kind::AssignmentExpr) {
      int accessID = iir::getAccessID(
          std::dynamic_pointer_cast<iir::AssignmentExpr>(exprStmt->getExpr())->getLeft());
      if(metadata.isAccessType(iir::FieldAccessType::LocalVariable, accessID)) {
        scalarAccessID = accessID;
      }
    }
  }

  if(scalarAccessID.has_value()) {
    DAWN_ASSERT_MSG(idToLocalVariableData.count(*scalarAccessID), "Uncategorized local variable.");
    if(idToLocalVariableData.at(*scalarAccessID).isScalar()) {
      return scalarAccessID;
    }
  }
  return std::nullopt;
}

// Returns the rhs `Expr` of a variable declaration / assignment.
std::shared_ptr<iir::Expr> getRhsOfAssignment(const std::shared_ptr<iir::Stmt> stmt) {
  if(stmt->getKind() == iir::Stmt::Kind::VarDeclStmt) {
    return std::dynamic_pointer_cast<iir::VarDeclStmt>(stmt)->getInitList()[0];
  } else if(stmt->getKind() == iir::Stmt::Kind::ExprStmt) {
    const auto& exprStmt = std::dynamic_pointer_cast<iir::ExprStmt>(stmt);
    if(exprStmt->getExpr()->getKind() == iir::Expr::Kind::AssignmentExpr) {
      return std::dynamic_pointer_cast<iir::AssignmentExpr>(exprStmt->getExpr())->getRight();
    }
  }

  dawn_unreachable("Function called with non-assignment stmt");
}

void removeScalarsFromDoMethod(iir::DoMethod& doMethod, iir::StencilMetaInformation& metadata) {

  // Map from variable's access id to last rhs assigned to the variable
  std::unordered_map<int, std::shared_ptr<const iir::Expr>> variableToLastRhsMap;

  for(auto stmtIt = doMethod.getAST().getStatements().begin();
      stmtIt != doMethod.getAST().getStatements().end();) {

    for(const auto& [varAccessID, lastRhs] : variableToLastRhsMap) {

      VarAccessesReplacer replacer(varAccessID, lastRhs);
      // Only apply replacer on the rhs if current statement is an assignment, otherwise apply on
      // the whole statement.
      if(const std::shared_ptr<iir::ExprStmt> exprStmt =
             std::dynamic_pointer_cast<iir::ExprStmt>(*stmtIt)) {
        if(const std::shared_ptr<iir::AssignmentExpr> assignmentExpr =
               std::dynamic_pointer_cast<iir::AssignmentExpr>(exprStmt->getExpr())) {

          assignmentExpr->getRight() = assignmentExpr->getRight()->acceptAndReplace(replacer);
          continue;
        }
      }
      (*stmtIt)->acceptAndReplace(replacer);
    }
    // TODO: var *= var case must be fixed
    // TODO: missing: scalars declared inside if-else construct

    auto scalarAccessID = getScalarWrittenByStatement(*stmtIt, metadata);
    if(scalarAccessID.has_value()) { // Writing to a scalar variable
      variableToLastRhsMap[*scalarAccessID] = getRhsOfAssignment(*stmtIt);
      stmtIt = doMethod.getAST().erase(stmtIt);
    } else {
      ++stmtIt;
    }
  }
  // Recompute the accesses metadata of all statements (removed variables means removed accesses)
  computeAccesses(metadata, doMethod.getAST().getStatements());
  doMethod.update(iir::NodeUpdateType::level);
  // Metadata cleanup
  for(const auto& pair : variableToLastRhsMap) {
    int scalarAccessID = pair.first;
    metadata.removeAccessID(scalarAccessID);
  }
}
} // namespace

bool PassRemoveScalars::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
    for(const auto& doMethod : iterateIIROver<iir::DoMethod>(*stencilPtr)) {
      // Local variables are local to a DoMethod. Remove scalar local variables from the statements
      // in this DoMethod.
      removeScalarsFromDoMethod(*doMethod, stencilInstantiation->getMetaData());
    }
  }
  return true;
}

} // namespace dawn
