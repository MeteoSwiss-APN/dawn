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
#include "dawn/Support/Logging.h"

#include <tuple>
#include <unordered_map>

namespace dawn {
namespace {

// TODO: If we had a function such as getDimensions(expr) = Scalar | OnEdges | ...
//       that would consider the dimensionality of local variables and fields
//       this checker would not be needed.
class ExprScalarChecker : public ast::ASTVisitorForwarding {
  const iir::StencilMetaInformation& metadata_;
  bool isScalar_ = true;

public:
  void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) override { isScalar_ = false; }
  void visit(const std::shared_ptr<iir::VarAccessExpr>& expr) override {
    if(expr->isLocal()) {
      isScalar_ &= metadata_.getLocalVariableDataFromAccessID(iir::getAccessID(expr)).isScalar();
    }
  }

  bool isScalar() const { return isScalar_; }

  ExprScalarChecker(const iir::StencilMetaInformation& metadata) : metadata_(metadata) {}
};

bool isExprScalar(const std::shared_ptr<iir::Expr>& expr,
                  const iir::StencilMetaInformation& metadata) {
  ExprScalarChecker checker(metadata);
  expr->accept(checker);
  return checker.isScalar();
}

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

  if(const auto& varDeclStmt = std::dynamic_pointer_cast<iir::VarDeclStmt>(stmt)) {
    scalarAccessID = iir::getAccessID(varDeclStmt);
  } else if(const auto& exprStmt = std::dynamic_pointer_cast<iir::ExprStmt>(stmt)) {
    if(const auto& assignmentExpr =
           std::dynamic_pointer_cast<iir::AssignmentExpr>(exprStmt->getExpr())) {
      int accessID = iir::getAccessID(assignmentExpr->getLeft());
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
  if(const auto& varDeclStmt = std::dynamic_pointer_cast<iir::VarDeclStmt>(stmt)) {
    return varDeclStmt->getInitList()[0];
  } else if(const auto& exprStmt = std::dynamic_pointer_cast<iir::ExprStmt>(stmt)) {
    if(const auto& assignmentExpr =
           std::dynamic_pointer_cast<iir::AssignmentExpr>(exprStmt->getExpr())) {

      return assignmentExpr->getRight();
    }
  }

  dawn_unreachable("Function called with non-assignment stmt");
}

void removeScalarsFromBlockStmt(
    iir::BlockStmt& blockStmt,
    std::unordered_map<int, std::shared_ptr<const iir::Expr>>& scalarToLastRhsMap,
    const iir::StencilMetaInformation& metadata) {

  for(auto stmtIt = blockStmt.getStatements().begin(); stmtIt != blockStmt.getStatements().end();) {

    for(const auto& [varAccessID, lastRhs] : scalarToLastRhsMap) {

      VarAccessesReplacer replacer(varAccessID, lastRhs);
      // Only apply replacer on the rhs if current statement is an assignment.
      if(const std::shared_ptr<iir::ExprStmt> exprStmt =
             std::dynamic_pointer_cast<iir::ExprStmt>(*stmtIt)) {
        if(const std::shared_ptr<iir::AssignmentExpr> assignmentExpr =
               std::dynamic_pointer_cast<iir::AssignmentExpr>(exprStmt->getExpr())) {

          assignmentExpr->getRight() = assignmentExpr->getRight()->acceptAndReplace(replacer);
        }
      }
    }
    // Need to treat if statements differently. There are block statements inside.
    if(const std::shared_ptr<iir::IfStmt> ifStmt =
           std::dynamic_pointer_cast<iir::IfStmt>(*stmtIt)) {

      DAWN_ASSERT_MSG(ifStmt->getThenStmt()->getKind() == iir::Stmt::Kind::BlockStmt,
                      "Then statement must be a block statement.");
      removeScalarsFromBlockStmt(*std::dynamic_pointer_cast<iir::BlockStmt>(ifStmt->getThenStmt()),
                                 scalarToLastRhsMap, metadata);

      if(ifStmt->hasElse()) {
        DAWN_ASSERT_MSG(ifStmt->getElseStmt()->getKind() == iir::Stmt::Kind::BlockStmt,
                        "Else statement must be a block statement.");
        removeScalarsFromBlockStmt(
            *std::dynamic_pointer_cast<iir::BlockStmt>(ifStmt->getElseStmt()), scalarToLastRhsMap,
            metadata);
      }

    } else { // Not an if statement
      auto scalarAccessID = getScalarWrittenByStatement(*stmtIt, metadata);
      if(scalarAccessID.has_value()) { // Writing to a scalar variable
        scalarToLastRhsMap[*scalarAccessID] = getRhsOfAssignment(*stmtIt);
        stmtIt = blockStmt.erase(stmtIt);
        continue; // to next statement
      }
    }
    ++stmtIt;
  }
}

void removeScalarsFromDoMethod(iir::DoMethod& doMethod, iir::StencilMetaInformation& metadata) {

  // Map from variable's access id to last rhs assigned to the variable
  std::unordered_map<int, std::shared_ptr<const iir::Expr>> scalarToLastRhsMap;

  removeScalarsFromBlockStmt(doMethod.getAST(), scalarToLastRhsMap, metadata);

  // Recompute the accesses metadata of all statements (removed variables means removed
  // accesses)
  computeAccesses(metadata, doMethod.getAST().getStatements());
  // Metadata cleanup
  for(const auto& pair : scalarToLastRhsMap) {
    int scalarAccessID = pair.first;
    metadata.removeAccessID(scalarAccessID);
  }
}
} // namespace

bool isStatementUnsupported(const std::shared_ptr<iir::Stmt>& stmt,
                            const iir::StencilMetaInformation& metadata) {
  if(const auto& exprStmt = std::dynamic_pointer_cast<iir::ExprStmt>(stmt)) {
    if(const auto& assignmentExpr =
           std::dynamic_pointer_cast<iir::AssignmentExpr>(exprStmt->getExpr())) {
      if(assignmentExpr->getOp() != "=") { // Compound assignment
        return true;
      }
    } else if(exprStmt->getExpr()->getKind() ==
              iir::Expr::Kind::UnaryOperator) { // Increment / decrement ops
      return true;
    }
  } else if(const std::shared_ptr<iir::IfStmt> ifStmt =
                std::dynamic_pointer_cast<iir::IfStmt>(stmt)) {
    if(isExprScalar(ifStmt->getCondExpr(), metadata)) {
      return true;
    }
    for(const auto& subStmt : ifStmt->getChildren()) {
      if(isStatementUnsupported(subStmt, metadata)) {
        return true;
      }
    }
  }

  return false;
}

bool PassRemoveScalars::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  // Check if we have unsupported statements. If we do, warn the user and skip the pass execution.
  for(const auto& stmt : iterateIIROverStmt(*stencilInstantiation->getIIR())) {
    if(isStatementUnsupported(stmt, stencilInstantiation->getMetaData())) {
      DAWN_LOG(WARNING) << "Unsupported statement at line " << stmt->getSourceLocation()
                        << ". Skipping removal of scalar variables.";
      return true;
    }
  }
  for(const auto& doMethod : iterateIIROver<iir::DoMethod>(*stencilInstantiation->getIIR())) {
    // Local variables are local to a DoMethod. Remove scalar local variables from the statements
    // in this DoMethod.
    removeScalarsFromDoMethod(*doMethod, stencilInstantiation->getMetaData());
  }
  return true;
}

} // namespace dawn
