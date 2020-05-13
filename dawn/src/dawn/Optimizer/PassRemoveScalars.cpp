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
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Support/Logger.h"

#include <tuple>
#include <unordered_map>

namespace dawn {
namespace {

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

// If `stmt` is a variable declaration of / an assignment to a scalar local variable,
// returns the access ID of the scalar. Otherwise returns std::nullopt.
std::optional<int> getScalarLhsOfStatement(const std::shared_ptr<iir::Stmt> stmt,
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

// Whether a `stmt` is an assignment or a variable declaration with non-empty init list.
bool hasRhs(const std::shared_ptr<const iir::Stmt> stmt) {
  if(const auto& varDeclStmt = std::dynamic_pointer_cast<const iir::VarDeclStmt>(stmt)) {
    return varDeclStmt->getInitList().size() == 1 && varDeclStmt->getInitList()[0] != nullptr;
  } else if(const auto& exprStmt = std::dynamic_pointer_cast<const iir::ExprStmt>(stmt)) {
    if(exprStmt->getExpr()->getKind() == iir::Expr::Kind::AssignmentExpr) {
      return true;
    }
  }
  return false;
}

// Returns the rhs `Expr` of a variable declaration / assignment.
std::shared_ptr<iir::Expr>& getRhs(const std::shared_ptr<iir::Stmt> stmt) {
  if(const auto& varDeclStmt = std::dynamic_pointer_cast<iir::VarDeclStmt>(stmt)) {
    DAWN_ASSERT(varDeclStmt->getInitList().size() == 1);
    return varDeclStmt->getInitList()[0];
  } else if(const auto& exprStmt = std::dynamic_pointer_cast<iir::ExprStmt>(stmt)) {
    if(const auto& assignmentExpr =
           std::dynamic_pointer_cast<iir::AssignmentExpr>(exprStmt->getExpr())) {
      return assignmentExpr->getRight();
    }
  }

  dawn_unreachable("Function called with invalid stmt");
}

// Removes scalar local variables from a BlockStmt and returns their access ids.
std::set<int> removeScalarsFromBlockStmt(
    iir::BlockStmt& blockStmt,
    std::unordered_map<int, std::shared_ptr<const iir::Expr>>& scalarToLastRhsMap,
    const iir::StencilMetaInformation& metadata) {

  std::set<int> removedScalarsIDs;

  for(auto stmtIt = blockStmt.getStatements().begin(); stmtIt != blockStmt.getStatements().end();) {

    for(const auto& [varAccessID, lastRhs] : scalarToLastRhsMap) {

      VarAccessesReplacer replacer(varAccessID, lastRhs);
      // Apply replacer on the rhs if current statement is an assignment / var decl
      if(hasRhs(*stmtIt)) {
        auto& rhs = getRhs(*stmtIt);
        rhs = rhs->acceptAndReplace(replacer);
      } else if(const std::shared_ptr<iir::IfStmt> ifStmt = std::dynamic_pointer_cast<iir::IfStmt>(
                    *stmtIt)) { // or on the condition expression if current statement is an if.
        ifStmt->getCondExpr() = ifStmt->getCondExpr()->acceptAndReplace(replacer);
      }
    }
    // Now process then and else block statements of an if
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
      auto scalarAccessID = getScalarLhsOfStatement(*stmtIt, metadata);
      if(scalarAccessID.has_value()) { // Writing to / declaring a scalar variable
        if(hasRhs(*stmtIt)) {
          scalarToLastRhsMap[*scalarAccessID] = getRhs(*stmtIt);
        }
        stmtIt = blockStmt.erase(stmtIt);
        removedScalarsIDs.insert(*scalarAccessID);
        continue; // to next statement
      }
    }
    ++stmtIt;
  }

  return removedScalarsIDs;
}

std::vector<std::string> removeScalarsFromDoMethod(iir::DoMethod& doMethod,
                                                   iir::StencilMetaInformation& metadata) {

  std::vector<std::string> removedScalars;
  // Map from scalar variable's access id to last rhs assigned to the variable.
  std::unordered_map<int, std::shared_ptr<const iir::Expr>> scalarToLastRhsMap;

  auto removedIDs = removeScalarsFromBlockStmt(doMethod.getAST(), scalarToLastRhsMap, metadata);

  // Recompute the accesses metadata of all statements (removed variables means removed
  // accesses)
  computeAccesses(metadata, doMethod.getAST().getStatements());
  // Collect names of removed scalars and cleanup metadata
  for(int scalarAccessID : removedIDs) {
    removedScalars.push_back(metadata.getNameFromAccessID(scalarAccessID));
    metadata.removeAccessID(scalarAccessID);
  }
  return removedScalars;
}

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
} // namespace

bool PassRemoveScalars::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  // Check if we have unsupported statements. If we do, warn the user and skip the pass execution.
  for(const auto& stmt : iterateIIROverStmt(*stencilInstantiation->getIIR())) {
    if(isStatementUnsupported(stmt, stencilInstantiation->getMetaData())) {
      DAWN_DIAG(INFO, stencilInstantiation->getMetaData().getFileName(), stmt->getSourceLocation())
          << "Unsupported statement. Skipping removal of scalar variables...";
      return true;
    }
  }
  for(const auto& doMethod : iterateIIROver<iir::DoMethod>(*stencilInstantiation->getIIR())) {
    // Local variables are local to a DoMethod. Remove scalar local variables from the statements
    // and metadata in this DoMethod.
    auto removedScalars = removeScalarsFromDoMethod(*doMethod, stencilInstantiation->getMetaData());
    if(context_.getOptions().ReportPassRemoveScalars) {
      for(const auto& varName : removedScalars) {
        DAWN_LOG(INFO) << stencilInstantiation->getName() << ": DoMethod: " << doMethod->getID()
                       << " removed variable: " << varName;
      }
    }
    // Recompute extents of fields
    doMethod->update(iir::NodeUpdateType::level);
  }
  return true;
}

} // namespace dawn
