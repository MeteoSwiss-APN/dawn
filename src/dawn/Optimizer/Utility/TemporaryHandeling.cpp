#include "dawn/Optimizer/Utility/TemporaryHandeling.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/MetaInformation.h"
#include "dawn/Optimizer/Replacing.h"
namespace dawn {

namespace {

class NameGetter : public ASTVisitorForwarding {
  const iir::IIR* iir_;
  const int AccessID_;
  const bool captureLocation_;

  std::string name_;
  std::vector<SourceLocation> locations_;

public:
  NameGetter(const iir::IIR* iir, int AccessID, bool captureLocation)
      : iir_(iir), AccessID_(AccessID), captureLocation_(captureLocation) {}

  virtual void visit(const std::shared_ptr<VarDeclStmt>& stmt) override {
    if(iir_->getMetaData()->getAccessIDFromStmt(stmt) == AccessID_) {
      name_ = stmt->getName();
      if(captureLocation_)
        locations_.push_back(stmt->getSourceLocation());
    }

    for(const auto& expr : stmt->getInitList())
      expr->accept(*this);
  }

  void visit(const std::shared_ptr<VarAccessExpr>& expr) override {
    if(iir_->getMetaData()->getAccessIDFromExpr(expr) == AccessID_) {
      name_ = expr->getName();
      if(captureLocation_)
        locations_.push_back(expr->getSourceLocation());
    }
  }

  void visit(const std::shared_ptr<LiteralAccessExpr>& expr) override {
    if(iir_->getMetaData()->getAccessIDFromExpr(expr) == AccessID_) {
      name_ = expr->getValue();
      if(captureLocation_)
        locations_.push_back(expr->getSourceLocation());
    }
  }

  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) override {
    if(iir_->getMetaData()->getAccessIDFromExpr(expr) == AccessID_) {
      name_ = expr->getName();
      if(captureLocation_)
        locations_.push_back(expr->getSourceLocation());
    }
  }

  std::pair<std::string, std::vector<SourceLocation>> getNameLocationPair() const {
    return std::make_pair(name_, locations_);
  }

  bool hasName() const { return !name_.empty(); }
  std::string getName() const { return name_; }
};

} // anonymous namespace

void promoteLocalVariableToTemporaryField(iir::IIR* iir, iir::Stencil* stencil, int AccessID,
                                          const iir::Stencil::Lifetime& lifetime) {
  std::string varname = iir->getMetaData()->getNameFromAccessID(AccessID);
  std::string fieldname = iir::StencilMetaInformation::makeTemporaryFieldname(
      iir::StencilMetaInformation::extractLocalVariablename(varname), AccessID);
  //  std::cout << "=======================================\nwe are trying to replace var : " <<
  //  varname
  //            << " with ID: " << AccessID << " with a temp-field\n\n"
  //            << std::endl;
  //  std::cout << "the size of the map is: " << iir->getMetaData()->getExprToAccessIDMap().size()
  //            << std::endl;
  //  std::cout << std::endl;

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

void renameAllOccurrences(iir::IIR* iir, iir::Stencil* stencil, int oldAccessID, int newAccessID) {
  // Rename the statements and accesses
  stencil->renameAllOccurrences(oldAccessID, newAccessID);

  // Remove form all AccessID maps
  iir->getMetaData()->removeAccessID(oldAccessID);
}

std::string getOriginalNameFromAccessID(int AccessID, const std::unique_ptr<iir::IIR>& iir) {
  NameGetter orignalNameGetter(iir.get(), AccessID, true);

  for(const auto& stmtAccessesPair : iterateIIROver<iir::StatementAccessesPair>(*iir)) {
    stmtAccessesPair->getStatement()->ASTStmt->accept(orignalNameGetter);
    if(orignalNameGetter.hasName())
      return orignalNameGetter.getName();
  }

  // Best we can do...
  return iir->getMetaData()->getNameFromAccessID(AccessID);
}

} // namespace dawn
