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
//===------------------------------------------------------------------------------------------===/

#include "dawn/Optimizer/TemporaryHandeling.h"
#include "dawn/IIR/InstantiationHelper.h"
#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/Replacing.h"
#include "dawn/Support/Assert.h"

namespace dawn {
void promoteLocalVariableToTemporaryField(iir::StencilInstantiation* instantiation,
                                          iir::Stencil* stencil, int accessID,
                                          const iir::Stencil::Lifetime& lifetime,
                                          iir::TemporaryScope temporaryScope) {
  std::string varname = instantiation->getMetaData().getFieldNameFromAccessID(accessID);
  std::string fieldname = iir::InstantiationHelper::makeTemporaryFieldname(
      iir::InstantiationHelper::extractLocalVariablename(varname), accessID);

  // Replace all variable accesses with field accesses
  stencil->forEachStatementAccessesPair(
      [&](ArrayRef<std::unique_ptr<iir::StatementAccessesPair>> statementAccessesPair) -> void {
        replaceVarWithFieldAccessInStmts(instantiation->getMetaData(), stencil, accessID, fieldname,
                                         statementAccessesPair);
      },
      lifetime);

  // Replace the the variable declaration with an assignment to the temporary field
  const std::vector<std::unique_ptr<iir::StatementAccessesPair>>& statementAccessesPairs =
      stencil->getStage(lifetime.Begin.StagePos)
          ->getChildren()
          .at(lifetime.Begin.DoMethodIndex)
          ->getChildren();
  std::shared_ptr<Statement> oldStatement =
      statementAccessesPairs[lifetime.Begin.StatementIndex]->getStatement();

  // The oldStmt has to be a `VarDeclStmt`. For example
  //
  //   double __local_foo = ...
  //
  // will be replaced with
  //
  //   __tmp_foo(0, 0, 0) = ...
  //
  iir::VarDeclStmt* varDeclStmt = dyn_cast<iir::VarDeclStmt>(oldStatement->ASTStmt.get());
  // If the TemporaryScope is within this stencil, then a VarDecl should be found (otherwise we have
  // a bug)
  DAWN_ASSERT_MSG((varDeclStmt || temporaryScope == iir::TemporaryScope::TS_Field),
                  format("Promote local variable to temporary field: a var decl is not "
                         "found for accessid: %i , name :%s",
                         accessID, instantiation->getMetaData().getNameFromAccessID(accessID))
                      .c_str());
  // If a vardecl is found, then during the promotion we would like to replace it as an assignment
  // statement to a field expression
  // Otherwise, the vardecl is in a different stencil (which is legal) therefore we take no action
  if(varDeclStmt) {
    DAWN_ASSERT_MSG(!varDeclStmt->isArray(), "cannot promote local array to temporary field");

    auto fieldAccessExpr = std::make_shared<iir::FieldAccessExpr>(fieldname);
    instantiation->getMetaData().insertExprToAccessID(fieldAccessExpr, accessID);
    auto assignmentExpr =
        std::make_shared<iir::AssignmentExpr>(fieldAccessExpr, varDeclStmt->getInitList().front());
    auto exprStmt = std::make_shared<iir::ExprStmt>(assignmentExpr);

    // Replace the statement
    statementAccessesPairs[lifetime.Begin.StatementIndex]->setStatement(
        std::make_shared<Statement>(exprStmt, oldStatement->StackTrace));

    // Remove the variable
    instantiation->getMetaData().removeAccessID(accessID);
    instantiation->getMetaData().eraseStmtToAccessID(oldStatement->ASTStmt);
  }
  // Register the field
  instantiation->getMetaData().insertAccessOfType(iir::FieldAccessType::FAT_StencilTemporary,
                                                  accessID, fieldname);

  // Update the fields of the stages we modified
  stencil->updateFields(lifetime);
}

void promoteTemporaryFieldToAllocatedField(iir::StencilInstantiation* instantiation, int AccessID) {
  DAWN_ASSERT(instantiation->getMetaData().isAccessType(iir::FieldAccessType::FAT_StencilTemporary,
                                                        AccessID));
  instantiation->getMetaData().moveRegisteredFieldTo(
      iir::FieldAccessType::FAT_InterStencilTemporary, AccessID);
}

void demoteTemporaryFieldToLocalVariable(iir::StencilInstantiation* instantiation,
                                         iir::Stencil* stencil, int AccessID,
                                         const iir::Stencil::Lifetime& lifetime) {
  std::string fieldname = instantiation->getMetaData().getFieldNameFromAccessID(AccessID);
  std::string varname = iir::InstantiationHelper::makeLocalVariablename(
      iir::InstantiationHelper::extractTemporaryFieldname(fieldname), AccessID);

  // Replace all field accesses with variable accesses
  stencil->forEachStatementAccessesPair(
      [&](ArrayRef<std::unique_ptr<iir::StatementAccessesPair>> statementAccessesPairs) -> void {
        replaceFieldWithVarAccessInStmts(instantiation->getMetaData(), stencil, AccessID, varname,
                                         statementAccessesPairs);
      },
      lifetime);

  // Replace the first access to the field with a VarDeclStmt
  const std::vector<std::unique_ptr<iir::StatementAccessesPair>>& statementAccessesPairs =
      stencil->getStage(lifetime.Begin.StagePos)
          ->getChildren()
          .at(lifetime.Begin.DoMethodIndex)
          ->getChildren();
  std::shared_ptr<Statement> oldStatement =
      statementAccessesPairs[lifetime.Begin.StatementIndex]->getStatement();

  // The oldStmt has to be an `ExprStmt` with an `AssignmentExpr`. For example
  //
  //   __tmp_foo(0, 0, 0) = ...
  //
  // will be replaced with
  //
  //   double __local_foo = ...
  //
  iir::ExprStmt* exprStmt = dyn_cast<iir::ExprStmt>(oldStatement->ASTStmt.get());
  DAWN_ASSERT_MSG(exprStmt, "first access of field (i.e lifetime.Begin) is not an `ExprStmt`");
  iir::AssignmentExpr* assignmentExpr = dyn_cast<iir::AssignmentExpr>(exprStmt->getExpr().get());
  DAWN_ASSERT_MSG(assignmentExpr,
                  "first access of field (i.e lifetime.Begin) is not an `AssignmentExpr`");

  // Create the new `VarDeclStmt` which will replace the old `ExprStmt`
  std::shared_ptr<iir::Stmt> varDeclStmt = std::make_shared<iir::VarDeclStmt>(
      Type(BuiltinTypeID::Float), varname, 0, "=",
      std::vector<std::shared_ptr<iir::Expr>>{assignmentExpr->getRight()});

  // Replace the statement
  statementAccessesPairs[lifetime.Begin.StatementIndex]->setStatement(
      std::make_shared<Statement>(varDeclStmt, oldStatement->StackTrace));

  // Remove the field
  instantiation->getMetaData().removeAccessID(AccessID);

  // Register the variable
  instantiation->getMetaData().setAccessIDNamePair(AccessID, varname);
  instantiation->getMetaData().insertStmtToAccessID(varDeclStmt, AccessID);

  // Update the fields of the stages we modified
  stencil->updateFields(lifetime);
}

} // namespace dawn