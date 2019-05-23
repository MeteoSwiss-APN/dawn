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

#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/InstantiationHelper.h"
#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/Optimizer/AccessComputation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassTemporaryType.h"
#include "dawn/Optimizer/Renaming.h"
#include "dawn/Optimizer/Replacing.h"
#include "dawn/Optimizer/StatementMapper.h"
#include "dawn/SIR/AST.h"
#include "dawn/SIR/ASTUtil.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/FileUtil.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/Json.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/Printing.h"
#include "dawn/Support/RemoveIf.hpp"
#include "dawn/Support/Twine.h"
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <stack>

namespace dawn {
namespace iir {

//===------------------------------------------------------------------------------------------===//
//     StencilInstantiation
//===------------------------------------------------------------------------------------------===//

StencilInstantiation::StencilInstantiation(dawn::OptimizerContext* context)
    : context_(context), metadata_(*(context->getSIR()->GlobalVariableMap)),
      IIR_(make_unique<IIR>(*(context->getSIR()->GlobalVariableMap),
                            context->getSIR()->StencilFunctions)) {}

StencilMetaInformation& StencilInstantiation::getMetaData() { return metadata_; }

std::shared_ptr<StencilInstantiation> StencilInstantiation::clone() const {

  std::shared_ptr<StencilInstantiation> stencilInstantiation =
      std::make_shared<StencilInstantiation>(context_);

  stencilInstantiation->metadata_.clone(metadata_);

  stencilInstantiation->IIR_ =
      make_unique<iir::IIR>(stencilInstantiation->getIIR()->getGlobalVariableMap(),
                            stencilInstantiation->getIIR()->getStencilFunctions());
  IIR_->clone(stencilInstantiation->IIR_);

  return stencilInstantiation;
}

const std::string StencilInstantiation::getName() const { return metadata_.getStencilName(); }

bool StencilInstantiation::insertBoundaryConditions(std::string originalFieldName,
                                                    std::shared_ptr<BoundaryConditionDeclStmt> bc) {
  if(metadata_.hasFieldBC(originalFieldName) != 0) {
    return false;
  } else {
    metadata_.insertFieldBC(originalFieldName, bc);
    return true;
  }
}

const sir::Value& StencilInstantiation::getGlobalVariableValue(const std::string& name) const {
  DAWN_ASSERT(IIR_->getGlobalVariableMap().count(name));
  return *(IIR_->getGlobalVariableMap().at(name));
}

int StencilInstantiation::createVersionAndRename(int AccessID, Stencil* stencil, int curStageIdx,
                                                 int curStmtIdx, std::shared_ptr<Expr>& expr,
                                                 RenameDirection dir) {

  int newAccessID = -1;
  if(metadata_.isAccessType(FieldAccessType::FAT_Field, AccessID)) {
    if(metadata_.getFieldAccessMetadata().variableVersions_.hasVariableMultipleVersions(AccessID)) {
      // Field is already multi-versioned, append a new version
      auto versions = metadata_.getFieldAccessMetadata().variableVersions_.getVersions(AccessID);

      // Set the second to last field to be a temporary (only the first and the last field will be
      // real storages, all other versions will be temporaries)
      int lastAccessID = versions->back();
      metadata_.moveRegisteredFieldTo(iir::FieldAccessType::FAT_StencilTemporary, lastAccessID);

      // The field with version 0 contains the original name
      const std::string& originalName = metadata_.getFieldNameFromAccessID(versions->front());

      // Register the new field
      newAccessID =
          metadata_.insertAccessOfType(iir::FieldAccessType::FAT_InterStencilTemporary,
                                       originalName + "_" + std::to_string(versions->size()));

      versions->push_back(newAccessID);
      metadata_.insertVersions(newAccessID, versions);

    } else {
      const std::string& originalName = metadata_.getFieldNameFromAccessID(AccessID);

      newAccessID = metadata_.insertAccessOfType(iir::FieldAccessType::FAT_InterStencilTemporary,
                                                 originalName + "_1");

      // Register the new *and* old field as being multi-versioned and indicate code-gen it has to
      // allocate the second version
      auto versionsVecPtr = std::make_shared<std::vector<int>>();
      *versionsVecPtr = {AccessID, newAccessID};

      metadata_.insertVersions(AccessID, versionsVecPtr);
      metadata_.insertVersions(newAccessID, versionsVecPtr);
    }
  } else {
    // if not a field, it is a variable
    if(metadata_.hasVariableMultipleVersions(AccessID)) {
      // Variable is already multi-versioned, append a new version
      auto versions = metadata_.getVersionsOf(AccessID);

      // The variable with version 0 contains the original name
      const std::string& originalName = metadata_.getFieldNameFromAccessID(versions->front());

      // Register the new variable
      newAccessID =
          metadata_.insertAccessOfType(iir::FieldAccessType::FAT_LocalVariable,
                                       originalName + "_" + std::to_string(versions->size()));
      versions->push_back(newAccessID);
      metadata_.insertVersions(newAccessID, versions);

    } else {
      const std::string& originalName = metadata_.getFieldNameFromAccessID(AccessID);

      newAccessID = metadata_.insertAccessOfType(iir::FieldAccessType::FAT_LocalVariable,
                                                 originalName + "_1");
      // Register the new *and* old variable as being multi-versioned
      auto versionsVecPtr = std::make_shared<std::vector<int>>();
      *versionsVecPtr = {AccessID, newAccessID};

      metadata_.insertVersions(AccessID, versionsVecPtr);
      metadata_.insertVersions(newAccessID, versionsVecPtr);
    }
  }

  // Rename the Expression
  renameAccessIDInExpr(this, AccessID, newAccessID, expr);

  // Recompute the accesses of the current statement (only works with single Do-Methods - for now)
  computeAccesses(this,
                  stencil->getStage(curStageIdx)->getSingleDoMethod().getChildren()[curStmtIdx]);

  // Rename the statement and accesses
  for(int stageIdx = curStageIdx;
      dir == RD_Above ? (stageIdx >= 0) : (stageIdx < stencil->getNumStages());
      dir == RD_Above ? stageIdx-- : stageIdx++) {
    Stage& stage = *stencil->getStage(stageIdx);
    DoMethod& doMethod = stage.getSingleDoMethod();

    if(stageIdx == curStageIdx) {
      for(int i = dir == RD_Above ? (curStmtIdx - 1) : (curStmtIdx + 1);
          dir == RD_Above ? (i >= 0) : (i < doMethod.getChildren().size());
          dir == RD_Above ? (--i) : (++i)) {
        renameAccessIDInStmts(&metadata_, AccessID, newAccessID, doMethod.getChildren()[i]);
        renameAccessIDInAccesses(&metadata_, AccessID, newAccessID, doMethod.getChildren()[i]);
      }

    } else {
      renameAccessIDInStmts(&metadata_, AccessID, newAccessID, doMethod.getChildren());
      renameAccessIDInAccesses(&metadata_, AccessID, newAccessID, doMethod.getChildren());
    }

    // Update the fields of the doMethod and stage levels
    doMethod.update(iir::NodeUpdateType::level);
    stage.update(iir::NodeUpdateType::level);
  }

  return newAccessID;
}

void StencilInstantiation::renameAllOccurrences(Stencil* stencil, int oldAccessID,
                                                int newAccessID) {
  // Rename the statements and accesses
  stencil->renameAllOccurrences(oldAccessID, newAccessID);

  // Remove form all AccessID maps
  metadata_.removeAccessID(oldAccessID);
}

bool StencilInstantiation::isIDAccessedMultipleStencils(int accessID) const {

  int count = 0;
  for(const auto& stencil : IIR_->getChildren()) {
    if(stencil->hasFieldAccessID(accessID)) {
      if(++count > 1)
        return true;
    }
  }
  return false;
}

void StencilInstantiation::promoteLocalVariableToTemporaryField(Stencil* stencil, int accessID,
                                                                const Stencil::Lifetime& lifetime,
                                                                TemporaryScope temporaryScope) {
  std::string varname = metadata_.getFieldNameFromAccessID(accessID);
  std::string fieldname = InstantiationHelper::makeTemporaryFieldname(
      InstantiationHelper::extractLocalVariablename(varname), accessID);

  // Replace all variable accesses with field accesses
  stencil->forEachStatementAccessesPair(
      [&](ArrayRef<std::unique_ptr<StatementAccessesPair>> statementAccessesPair) -> void {
        replaceVarWithFieldAccessInStmts(metadata_, stencil, accessID, fieldname,
                                         statementAccessesPair);
      },
      lifetime);

  // Replace the the variable declaration with an assignment to the temporary field
  const std::vector<std::unique_ptr<StatementAccessesPair>>& statementAccessesPairs =
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
  // If the TemporaryScope is within this stencil, then a VarDecl should be found (otherwise we have
  // a bug)
  DAWN_ASSERT_MSG((varDeclStmt || temporaryScope == TemporaryScope::TS_Field),
                  format("Promote local variable to temporary field: a var decl is not "
                         "found for accessid: %i , name :%s",
                         accessID, metadata_.getNameFromAccessID(accessID))
                      .c_str());
  // If a vardecl is found, then during the promotion we would like to replace it as an assignment
  // statement to a field expression
  // Otherwise, the vardecl is in a different stencil (which is legal) therefore we take no action
  if(varDeclStmt) {
    DAWN_ASSERT_MSG(!varDeclStmt->isArray(), "cannot promote local array to temporary field");

    auto fieldAccessExpr = std::make_shared<FieldAccessExpr>(fieldname);
    metadata_.insertExprToAccessID(fieldAccessExpr, accessID);
    auto assignmentExpr =
        std::make_shared<AssignmentExpr>(fieldAccessExpr, varDeclStmt->getInitList().front());
    auto exprStmt = std::make_shared<ExprStmt>(assignmentExpr);

    // Replace the statement
    statementAccessesPairs[lifetime.Begin.StatementIndex]->setStatement(
        std::make_shared<Statement>(exprStmt, oldStatement->StackTrace));

    // Remove the variable
    metadata_.removeAccessID(accessID);
    metadata_.eraseStmtToAccessID(oldStatement->ASTStmt);
  }
  // Register the field
  metadata_.insertAccessOfType(FieldAccessType::FAT_StencilTemporary, accessID, fieldname);

  // Update the fields of the stages we modified
  stencil->updateFields(lifetime);
}

void StencilInstantiation::promoteTemporaryFieldToAllocatedField(int AccessID) {
  DAWN_ASSERT(metadata_.isAccessType(iir::FieldAccessType::FAT_StencilTemporary, AccessID));
  metadata_.moveRegisteredFieldTo(FieldAccessType::FAT_InterStencilTemporary, AccessID);
}

void StencilInstantiation::demoteTemporaryFieldToLocalVariable(Stencil* stencil, int AccessID,
                                                               const Stencil::Lifetime& lifetime) {
  std::string fieldname = metadata_.getFieldNameFromAccessID(AccessID);
  std::string varname = InstantiationHelper::makeLocalVariablename(
      InstantiationHelper::extractTemporaryFieldname(fieldname), AccessID);

  // Replace all field accesses with variable accesses
  stencil->forEachStatementAccessesPair(
      [&](ArrayRef<std::unique_ptr<StatementAccessesPair>> statementAccessesPairs) -> void {
        replaceFieldWithVarAccessInStmts(metadata_, stencil, AccessID, varname,
                                         statementAccessesPairs);
      },
      lifetime);

  // Replace the first access to the field with a VarDeclStmt
  const std::vector<std::unique_ptr<StatementAccessesPair>>& statementAccessesPairs =
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
  metadata_.removeAccessID(AccessID);

  // Register the variable
  metadata_.setAccessIDNamePair(AccessID, varname);
  metadata_.insertStmtToAccessID(varDeclStmt, AccessID);

  // Update the fields of the stages we modified
  stencil->updateFields(lifetime);
}

std::shared_ptr<StencilFunctionInstantiation>
StencilInstantiation::makeStencilFunctionInstantiation(
    const std::shared_ptr<StencilFunCallExpr>& expr,
    const std::shared_ptr<sir::StencilFunction>& SIRStencilFun, const std::shared_ptr<AST>& ast,
    const Interval& interval,
    const std::shared_ptr<StencilFunctionInstantiation>& curStencilFunctionInstantiation) {

  std::shared_ptr<StencilFunctionInstantiation> stencilFun =
      std::make_shared<StencilFunctionInstantiation>(this, expr, SIRStencilFun, ast, interval,
                                                     curStencilFunctionInstantiation != nullptr);

  metadata_.insertStencilFunInstantiationCandidate(
      stencilFun, StencilMetaInformation::StencilFunctionInstantiationCandidate{
                      curStencilFunctionInstantiation});

  return stencilFun;
}

namespace {

/// @brief Get the orignal name of the field (or variable) given by AccessID and a list of
/// SourceLocations where this field (or variable) was accessed.
class OriginalNameGetter : public ASTVisitorForwarding {
  const StencilMetaInformation& metadata_;
  const int AccessID_;
  const bool captureLocation_;

  std::string name_;
  std::vector<SourceLocation> locations_;

public:
  OriginalNameGetter(const StencilMetaInformation& metadata, int AccessID, bool captureLocation)
      : metadata_(metadata), AccessID_(AccessID), captureLocation_(captureLocation) {}

  virtual void visit(const std::shared_ptr<VarDeclStmt>& stmt) override {
    if(metadata_.getAccessIDFromStmt(stmt) == AccessID_) {
      name_ = stmt->getName();
      if(captureLocation_)
        locations_.push_back(stmt->getSourceLocation());
    }

    for(const auto& expr : stmt->getInitList())
      expr->accept(*this);
  }

  void visit(const std::shared_ptr<VarAccessExpr>& expr) override {
    if(metadata_.getAccessIDFromExpr(expr) == AccessID_) {
      name_ = expr->getName();
      if(captureLocation_)
        locations_.push_back(expr->getSourceLocation());
    }
  }

  void visit(const std::shared_ptr<LiteralAccessExpr>& expr) override {
    if(metadata_.getAccessIDFromExpr(expr) == AccessID_) {
      name_ = expr->getValue();
      if(captureLocation_)
        locations_.push_back(expr->getSourceLocation());
    }
  }

  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) override {
    if(metadata_.getAccessIDFromExpr(expr) == AccessID_) {
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

std::pair<std::string, std::vector<SourceLocation>>
StencilInstantiation::getOriginalNameAndLocationsFromAccessID(
    int AccessID, const std::shared_ptr<Stmt>& stmt) const {
  OriginalNameGetter orignalNameGetter(metadata_, AccessID, true);
  stmt->accept(orignalNameGetter);
  return orignalNameGetter.getNameLocationPair();
}

std::string StencilInstantiation::getOriginalNameFromAccessID(int AccessID) const {
  OriginalNameGetter orignalNameGetter(metadata_, AccessID, true);

  for(const auto& stmtAccessesPair : iterateIIROver<StatementAccessesPair>(*getIIR())) {
    stmtAccessesPair->getStatement()->ASTStmt->accept(orignalNameGetter);
    if(orignalNameGetter.hasName())
      return orignalNameGetter.getName();
  }

  // Best we can do...
  return metadata_.getFieldNameFromAccessID(AccessID);
}

bool StencilInstantiation::checkTreeConsistency() const { return IIR_->checkTreeConsistency(); }

void StencilInstantiation::jsonDump(std::string filename) const {

  std::ofstream fs(filename, std::ios::out | std::ios::trunc);
  if(!fs.is_open()) {
    DiagnosticsBuilder diag(DiagnosticsKind::Error, SourceLocation());
    diag << "file system error: cannot open file: " << filename;
    context_->getDiagnostics().report(diag);
  }

  json::json node;
  node["MetaInformation"] = metadata_.jsonDump();
  node["IIR"] = IIR_->jsonDump();
  fs << node.dump(2) << std::endl;
  fs.close();
}

void StencilInstantiation::reportAccesses() const {
  // Stencil functions
  for(const auto& stencilFun : metadata_.getStencilFunctionInstantiations()) {
    const auto& statementAccessesPairs = stencilFun->getStatementAccessesPairs();

    for(std::size_t i = 0; i < statementAccessesPairs.size(); ++i) {
      std::cout << "\nACCESSES: line "
                << statementAccessesPairs[i]->getStatement()->ASTStmt->getSourceLocation().Line
                << ": "
                << statementAccessesPairs[i]->getCalleeAccesses()->reportAccesses(stencilFun.get())
                << "\n";
    }
  }

  // Stages

  for(const auto& stmtAccessesPair : iterateIIROver<StatementAccessesPair>(*getIIR())) {
    std::cout << "\nACCESSES: line "
              << stmtAccessesPair->getStatement()->ASTStmt->getSourceLocation().Line << ": "
              << stmtAccessesPair->getAccesses()->reportAccesses(metadata_) << "\n";
  }
}

} // namespace iir
} // namespace dawn
