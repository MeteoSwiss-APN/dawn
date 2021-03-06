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
#include "dawn/AST/LocationType.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTStmt.h"
#include "dawn/IIR/InstantiationHelper.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/Replacing.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Assert.h"

namespace dawn {

namespace {
ast::HorizontalFieldDimension
getHorizontalFieldDimensionFromVar(iir::StencilInstantiation const* instantiation, int accessID) {
  auto cartesian = instantiation->getIIR()->getGridType() == ast::GridType::Cartesian;
  if(cartesian) {
    return ast::HorizontalFieldDimension(ast::cartesian, {true, true});
  } else {
    auto locType =
        instantiation->getMetaData().getLocalVariableDataFromAccessID(accessID).getLocationType();
    return ast::HorizontalFieldDimension(ast::unstructured, locType);
  }
}
} // namespace

void promoteLocalVariableToTemporaryField(iir::StencilInstantiation* instantiation,
                                          iir::Stencil* stencil, int accessID,
                                          const iir::Stencil::Lifetime& lifetime,
                                          iir::TemporaryScope temporaryScope) {
  std::string varname = instantiation->getMetaData().getFieldNameFromAccessID(accessID);

  // Figure out dimensions
  ast::FieldDimensions fieldDims =
      ast::FieldDimensions{getHorizontalFieldDimensionFromVar(instantiation, accessID), true};

  // Compute name of field
  std::string fieldname = iir::InstantiationHelper::makeTemporaryFieldname(
      iir::InstantiationHelper::extractLocalVariablename(varname), accessID);

  // Replace all variable accesses with field accesses
  stencil->forEachStatement(
      [&](ArrayRef<std::shared_ptr<ast::Stmt>> stmt) -> void {
        replaceVarWithFieldAccessInStmts(stencil, accessID, fieldname, stmt);
      },
      lifetime);

  // Replace the the variable declaration with an assignment to the temporary field
  ast::BlockStmt& blockStmt = stencil->getStage(lifetime.Begin.StagePos)
                                  ->getChildren()
                                  .at(lifetime.Begin.DoMethodIndex)
                                  ->getAST();

  const std::shared_ptr<ast::Stmt> oldStatement =
      blockStmt.getStatements()[lifetime.Begin.StatementIndex];

  // The oldStmt has to be a `VarDeclStmt`. For example
  //
  //   double __local_foo = ...
  //
  // will be replaced with
  //
  //   __tmp_foo(0, 0, 0) = ...
  //
  ast::VarDeclStmt* varDeclStmt = dyn_cast<ast::VarDeclStmt>(oldStatement.get());
  // If the TemporaryScope is within this stencil, then a VarDecl should be found (otherwise we have
  // a bug)
  DAWN_ASSERT_MSG((varDeclStmt || instantiation->isIDAccessedMultipleStencils(accessID)),
                  format("Promote local variable to temporary field: a var decl is not "
                         "found for accessid: %i , name :%s",
                         accessID, instantiation->getMetaData().getNameFromAccessID(accessID))
                      .c_str());
  // If a vardecl is found, then during the promotion we would like to replace it as an assignment
  // statement to a field expression
  // Otherwise, the vardecl is in a different stencil (which is legal) therefore we take no action
  if(varDeclStmt) {
    DAWN_ASSERT_MSG(!varDeclStmt->isArray(), "cannot promote local array to temporary field");

    auto fieldAccessExpr = std::make_shared<ast::FieldAccessExpr>(fieldname);
    fieldAccessExpr->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(accessID);
    auto assignmentExpr =
        std::make_shared<ast::AssignmentExpr>(fieldAccessExpr, varDeclStmt->getInitList().front());
    auto exprStmt = iir::makeExprStmt(assignmentExpr);

    // Replace the statement
    exprStmt->getData<iir::IIRStmtData>() = std::move(oldStatement->getData<iir::IIRStmtData>());
    blockStmt.replaceChildren(oldStatement, exprStmt);

    // Remove the variable
    instantiation->getMetaData().removeAccessID(accessID);
  }

  // Register the field in the metadata
  instantiation->getMetaData().insertAccessOfType(iir::FieldAccessType::StencilTemporary, accessID,
                                                  fieldname);
  instantiation->getMetaData().setFieldDimensions(accessID, std::move(fieldDims));

  // Update the fields of the stages we modified
  stencil->updateFields(lifetime);
}

void promoteTemporaryFieldToAllocatedField(iir::StencilInstantiation* instantiation, int AccessID) {
  DAWN_ASSERT(
      instantiation->getMetaData().isAccessType(iir::FieldAccessType::StencilTemporary, AccessID));
  instantiation->getMetaData().moveRegisteredFieldTo(iir::FieldAccessType::InterStencilTemporary,
                                                     AccessID);
}

void demoteTemporaryFieldToLocalVariable(iir::StencilInstantiation* instantiation,
                                         iir::Stencil* stencil, int AccessID,
                                         const iir::Stencil::Lifetime& lifetime) {
  std::string fieldname = instantiation->getMetaData().getFieldNameFromAccessID(AccessID);
  std::string varname = iir::InstantiationHelper::makeLocalVariablename(
      iir::InstantiationHelper::extractTemporaryFieldname(fieldname), AccessID);

  // Replace all field accesses with variable accesses
  stencil->forEachStatement(
      [&](ArrayRef<std::shared_ptr<ast::Stmt>> stmts) -> void {
        replaceFieldWithVarAccessInStmts(stencil, AccessID, varname, stmts);
      },
      lifetime);

  // Replace the first access to the field with a VarDeclStmt
  ast::BlockStmt& blockStmt = stencil->getStage(lifetime.Begin.StagePos)
                                  ->getChildren()
                                  .at(lifetime.Begin.DoMethodIndex)
                                  ->getAST();
  const std::shared_ptr<ast::Stmt> oldStatement =
      blockStmt.getStatements()[lifetime.Begin.StatementIndex];

  // The oldStmt has to be an `ExprStmt` with an `AssignmentExpr`. For example
  //
  //   __tmp_foo(0, 0, 0) = ...
  //
  // will be replaced with
  //
  //   double __local_foo = ...
  //
  ast::ExprStmt* exprStmt = dyn_cast<ast::ExprStmt>(oldStatement.get());
  DAWN_ASSERT_MSG(
      exprStmt,
      "first access of temp field to be demoted (i.e lifetime.Begin) is not an `ExprStmt`");
  ast::AssignmentExpr* assignmentExpr = dyn_cast<ast::AssignmentExpr>(exprStmt->getExpr().get());
  DAWN_ASSERT_MSG(assignmentExpr, "first access of temp field to be demoted (i.e "
                                  "lifetime.Begin) is not an `AssignmentExpr`");

  // Remove the field
  instantiation->getMetaData().removeAccessID(AccessID);

  // Create the new `VarDeclStmt` which will replace the old `ExprStmt` and register the variable
  std::shared_ptr<ast::Stmt> varDeclStmt = instantiation->getMetaData().declareVar(
      true, varname, Type(BuiltinTypeID::Float), assignmentExpr->getRight(), AccessID);

  // Replace the statement
  varDeclStmt->getData<iir::IIRStmtData>().StackTrace =
      oldStatement->getData<iir::IIRStmtData>().StackTrace;
  varDeclStmt->getData<iir::IIRStmtData>().CallerAccesses =
      oldStatement->getData<iir::IIRStmtData>().CallerAccesses;
  varDeclStmt->getData<iir::IIRStmtData>().CalleeAccesses =
      oldStatement->getData<iir::IIRStmtData>().CalleeAccesses;
  blockStmt.replaceChildren(oldStatement, varDeclStmt);

  // Update the fields of the stages we modified
  stencil->updateFields(lifetime);
}

} // namespace dawn
