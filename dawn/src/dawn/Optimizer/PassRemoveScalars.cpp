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

// TODO: replace with a more general expr.getDimensions() = Scalar | OnEdges | ...
//      that considers the dimensionality of local variables and fields
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
      if(assignmentExpr->getOp() != "=") {
        throw std::runtime_error(dawn::format("Compound assignment not supported at line %d",
                                              assignmentExpr->getSourceLocation().Line));
      }
      return assignmentExpr->getRight();
    }
  }

  dawn_unreachable("Function called with non-assignment stmt");
}

/// TODO: if-statements with conditions containing only globals, scalars, literals (and not
/// dimensional variables or fields) are not supported yet. (If the condition expression has
/// dimensions instead, then each variable we write to, inside the then/else blocks, will have
/// dimensions)
///
/// Would need a new pass to run before this one (and after PassLocalVarType):
/// (recall that a global cannot be set inside a DoMethod)
///
/// if(scalar_expr) { // with scalar_expr containing only globals, scalars, literals
///    varA = varA + 5.0;
///    f_c = varB + varA;
/// } else {
///    varB = 2.0;
///    f_c = varA * varB;
/// }
///
/// CHANGE INTO
///
/// bool _ifCond_0 = scalar_expr;
/// varA = _ifCond_0 ? varA + 5.0 : varA;
/// f_c = _ifCond_0 ? varB + varA;
/// varB = !_ifCond_0 ? 2.0 : varB;
/// f_c = !_ifCond_0 ? varA * varB : f_c;
///
/// than this pass (PassRemoveScalars) will remove the scalars _ifCond_0, varA, varB, ...
/// another pass could also remove the ternary operators if the conditions are literals.
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

        } else if(exprStmt->getExpr()->getKind() != iir::Expr::Kind::StencilFunCallExpr) {
          throw std::runtime_error(dawn::format("Unsupported statement at line %d",
                                                exprStmt->getSourceLocation().Line)); // e.g. i++;
        }
      }
    }
    // Need to treat if statements differently. There are block statements inside.
    if(const std::shared_ptr<iir::IfStmt> ifStmt =
           std::dynamic_pointer_cast<iir::IfStmt>(*stmtIt)) {

      if(isExprScalar(ifStmt->getCondExpr(), metadata)) {
        throw std::runtime_error(
            dawn::format("If-condition is scalar at line %d. It is not yet supported.",
                         ifStmt->getSourceLocation().Line)); // See reason above.
      }

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

bool PassRemoveScalars::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  // TODO: sir from GTClang currently not supported: it introduces compound assignments (handling
  // them complicates this pass). For now disabling the pass for cartesian grids.
  // We should remove compound assignments and increment/decrement ops from SIR/IIR. They are
  // syntactic sugar, only useful at DSL level.
  if(stencilInstantiation->getIIR()->getGridType() == ast::GridType::Cartesian) {
    return true;
  }
  for(const auto& doMethod : iterateIIROver<iir::DoMethod>(*stencilInstantiation->getIIR())) {
    // Local variables are local to a DoMethod. Remove scalar local variables from the statements
    // in this DoMethod.
    removeScalarsFromDoMethod(*doMethod, stencilInstantiation->getMetaData());
  }
  return true;
}

} // namespace dawn
