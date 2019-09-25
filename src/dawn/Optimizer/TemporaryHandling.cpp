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

#include "dawn/Optimizer/TemporaryHandling.h"
#include "dawn/IIR/ASTStmt.h"
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
  std::shared_ptr<iir::Stmt> oldStatement =
      statementAccessesPairs[lifetime.Begin.StatementIndex]->getStatement();

  // The oldStmt has to be a `VarDeclStmt`. For example
  //
  //   double __local_foo = ...
  //
  // will be replaced with
  //
  //   __tmp_foo(0, 0, 0) = ...
  //
  iir::VarDeclStmt* varDeclStmt = dyn_cast<iir::VarDeclStmt>(oldStatement.get());
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
    auto exprStmt = iir::makeExprStmt(assignmentExpr);

    // Replace the statement
    exprStmt->getData<iir::IIRStmtData>().StackTrace =
        oldStatement->getData<iir::IIRStmtData>().StackTrace;
    exprStmt->getData<iir::IIRStmtData>().CallerAccesses =
        oldStatement->getData<iir::IIRStmtData>().CallerAccesses;
    exprStmt->getData<iir::IIRStmtData>().CalleeAccesses =
        oldStatement->getData<iir::IIRStmtData>().CalleeAccesses;
    statementAccessesPairs[lifetime.Begin.StatementIndex]->setStatement(exprStmt);

    // Remove the variable
    instantiation->getMetaData().removeAccessID(accessID);
    instantiation->getMetaData().eraseStmtToAccessID(oldStatement);
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
  std::shared_ptr<iir::Stmt> oldStatement =
      statementAccessesPairs[lifetime.Begin.StatementIndex]->getStatement();

  // The oldStmt has to be an `ExprStmt` with an `AssignmentExpr`. For example
  //
  //   __tmp_foo(0, 0, 0) = ...
  //
  // will be replaced with
  //
  //   double __local_foo = ...
  //
  iir::ExprStmt* exprStmt = dyn_cast<iir::ExprStmt>(oldStatement.get());
  DAWN_ASSERT_MSG(exprStmt, "first access of field (i.e lifetime.Begin) is not an `ExprStmt`");
  iir::AssignmentExpr* assignmentExpr = dyn_cast<iir::AssignmentExpr>(exprStmt->getExpr().get());
  DAWN_ASSERT_MSG(assignmentExpr,
                  "first access of field (i.e lifetime.Begin) is not an `AssignmentExpr`");

  // Create the new `VarDeclStmt` which will replace the old `ExprStmt`
  std::shared_ptr<iir::Stmt> varDeclStmt =
      iir::makeVarDeclStmt(Type(BuiltinTypeID::Float), varname, 0, "=",
                           std::vector<std::shared_ptr<iir::Expr>>{assignmentExpr->getRight()});

  // Replace the statement
  varDeclStmt->getData<iir::IIRStmtData>().StackTrace =
      oldStatement->getData<iir::IIRStmtData>().StackTrace;
  varDeclStmt->getData<iir::IIRStmtData>().CallerAccesses =
      oldStatement->getData<iir::IIRStmtData>().CallerAccesses;
  varDeclStmt->getData<iir::IIRStmtData>().CalleeAccesses =
      oldStatement->getData<iir::IIRStmtData>().CalleeAccesses;
  statementAccessesPairs[lifetime.Begin.StatementIndex]->setStatement(varDeclStmt);

  // Remove the field
  instantiation->getMetaData().removeAccessID(AccessID);

  // Register the variable
  instantiation->getMetaData().addAccessIDNamePair(AccessID, varname);
  instantiation->getMetaData().addStmtToAccessID(varDeclStmt, AccessID);

  // Update the fields of the stages we modified
  stencil->updateFields(lifetime);
}

} // namespace dawn
