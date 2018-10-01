#include "dawn/Optimizer/Utility/TemporaryHandeling.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/MetaInformation.h"
#include "dawn/Optimizer/Replacing.h"

namespace dawn {

void promoteLocalVariableToTemporaryField(iir::IIR* iir, iir::Stencil* stencil, int AccessID,
                                          const iir::Stencil::Lifetime& lifetime) {
  std::string varname = iir->getMetaData()->getNameFromAccessID(AccessID);
  std::string fieldname = iir::StencilMetaInformation::makeTemporaryFieldname(
      iir::StencilMetaInformation::extractLocalVariablename(varname), AccessID);

  // Replace all variable accesses with field accesses
  stencil->forEachStatementAccessesPair(
      [&](ArrayRef<std::unique_ptr<iir::StatementAccessesPair>> statementAccessesPair) -> void {
        replaceVarWithFieldAccessInStmts(stencil, AccessID, fieldname, statementAccessesPair);
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
  VarDeclStmt* varDeclStmt = dyn_cast<VarDeclStmt>(oldStatement->ASTStmt.get());
  DAWN_ASSERT_MSG(varDeclStmt,
                  "first access to variable (i.e lifetime.Begin) is not an `VarDeclStmt`");
  DAWN_ASSERT_MSG(!varDeclStmt->isArray(), "cannot promote local array to temporary field");

  auto fieldAccessExpr = std::make_shared<FieldAccessExpr>(fieldname);
  iir->getMetaData()->getExprToAccessIDMap().emplace(fieldAccessExpr, AccessID);
  auto assignmentExpr =
      std::make_shared<AssignmentExpr>(fieldAccessExpr, varDeclStmt->getInitList().front());
  auto exprStmt = std::make_shared<ExprStmt>(assignmentExpr);

  // Replace the statement
  statementAccessesPairs[lifetime.Begin.StatementIndex]->setStatement(
      std::make_shared<Statement>(exprStmt, oldStatement->StackTrace));

  // Remove the variable
  iir->getMetaData()->removeAccessID(AccessID);
  iir->getMetaData()->getStmtToAccessIDMap().erase(oldStatement->ASTStmt);

  // Register the field
  iir->getMetaData()->setAccessIDNamePairOfField(AccessID, fieldname, true);

  // Update the fields of the stages we modified
  stencil->updateFields(lifetime);
}

void demoteTemporaryFieldToLocalVariable(iir::IIR* iir, iir::Stencil* stencil, int AccessID,
                                         const iir::Stencil::Lifetime& lifetime) {
  std::string fieldname = iir->getMetaData()->getNameFromAccessID(AccessID);
  std::string varname = iir->getMetaData()->makeLocalVariablename(
      iir->getMetaData()->extractTemporaryFieldname(fieldname), AccessID);

  // Replace all field accesses with variable accesses
  stencil->forEachStatementAccessesPair(
      [&](ArrayRef<std::unique_ptr<iir::StatementAccessesPair>> statementAccessesPairs) -> void {
        replaceFieldWithVarAccessInStmts(stencil, AccessID, varname, statementAccessesPairs);
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
  ExprStmt* exprStmt = dyn_cast<ExprStmt>(oldStatement->ASTStmt.get());
  DAWN_ASSERT_MSG(exprStmt, "first access of field (i.e lifetime.Begin) is not an `ExprStmt`");
  AssignmentExpr* assignmentExpr = dyn_cast<AssignmentExpr>(exprStmt->getExpr().get());
  DAWN_ASSERT_MSG(assignmentExpr,
                  "first access of field (i.e lifetime.Begin) is not an `AssignmentExpr`");

  // Create the new `VarDeclStmt` which will replace the old `ExprStmt`
  std::shared_ptr<Stmt> varDeclStmt =
      std::make_shared<VarDeclStmt>(Type(BuiltinTypeID::Float), varname, 0, "=",
                                    std::vector<std::shared_ptr<Expr>>{assignmentExpr->getRight()});

  // Replace the statement
  statementAccessesPairs[lifetime.Begin.StatementIndex]->setStatement(
      std::make_shared<Statement>(varDeclStmt, oldStatement->StackTrace));

  // Remove the field
  iir->getMetaData()->removeAccessID(AccessID);

  // Register the variable
  iir->getMetaData()->setAccessIDNamePair(AccessID, varname);
  iir->getMetaData()->getStmtToAccessIDMap().emplace(varDeclStmt, AccessID);

  // Update the fields of the stages we modified
  stencil->updateFields(lifetime);
}

void promoteTemporaryFieldToAllocatedField(iir::IIR* iir, int AccessID) {
  DAWN_ASSERT(iir->getMetaData()->isTemporaryField(AccessID));
  iir->getMetaData()->getTemporaryFieldAccessIDSet().erase(AccessID);
  iir->getMetaData()->getAllocatedFieldAccessIDSet().insert(AccessID);
}

void renameAllOccurrences(iir::IIR *iir, iir::Stencil *stencil, int oldAccessID, int newAccessID)
{
    // Rename the statements and accesses
    stencil->renameAllOccurrences(oldAccessID, newAccessID);

    // Remove form all AccessID maps
    iir->getMetaData()->removeAccessID(oldAccessID);
}

} // namespace dawn
