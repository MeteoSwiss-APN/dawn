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
    : context_(context), IIR_(make_unique<IIR>()) {}

StencilMetaInformation& StencilInstantiation::getMetaData() { return metadata_; }

std::shared_ptr<StencilInstantiation> StencilInstantiation::clone() const {

  std::shared_ptr<StencilInstantiation> stencilInstantiation =
      std::make_shared<StencilInstantiation>(context_);

  stencilInstantiation->metadata_.clone(metadata_);

  stencilInstantiation->IIR_ = make_unique<iir::IIR>();
  IIR_->clone(stencilInstantiation->IIR_);

  return stencilInstantiation;
}

void StencilInstantiation::setAccessIDNamePair(int AccessID, const std::string& name) {
  metadata_.AccessIDToNameMap_.emplace(AccessID, name);
  getNameToAccessIDMap().emplace(name, AccessID);
}

void StencilInstantiation::setAccessIDNamePairOfField(int AccessID, const std::string& name,
                                                      bool isTemporary) {
  setAccessIDNamePair(AccessID, name);
  metadata_.FieldAccessIDSet_.insert(AccessID);
  if(isTemporary) {
    metadata_.TemporaryFieldAccessIDSet_.insert(AccessID);
  }
}

void StencilInstantiation::setAccessIDNamePairOfGlobalVariable(int AccessID,
                                                               const std::string& name) {
  setAccessIDNamePair(AccessID, name);
  metadata_.GlobalVariableAccessIDSet_.insert(AccessID);
}

void StencilInstantiation::removeAccessID(int AccessID) {
  if(getNameToAccessIDMap().count(metadata_.AccessIDToNameMap_[AccessID]))
    getNameToAccessIDMap().erase(metadata_.AccessIDToNameMap_[AccessID]);

  metadata_.AccessIDToNameMap_.erase(AccessID);
  metadata_.FieldAccessIDSet_.erase(AccessID);
  metadata_.TemporaryFieldAccessIDSet_.erase(AccessID);

  if(metadata_.variableVersions_.hasVariableMultipleVersions(AccessID)) {
    auto versions = metadata_.variableVersions_.getVersions(AccessID);
    versions->erase(std::remove_if(versions->begin(), versions->end(),
                                   [&](int AID) { return AID == AccessID; }),
                    versions->end());
  }
}

const std::string StencilInstantiation::getName() const { return metadata_.stencilName_; }

std::unordered_map<int, int>& StencilInstantiation::getStmtIDToAccessIDMap() {
  return metadata_.StmtIDToAccessIDMap_;
}

const std::string& StencilInstantiation::getNameFromAccessID(int AccessID) const {
  if(AccessID < 0)
    return getNameFromLiteralAccessID(AccessID);
  auto it = metadata_.AccessIDToNameMap_.find(AccessID);
  DAWN_ASSERT_MSG(it != metadata_.AccessIDToNameMap_.end(), "Invalid AccessID");
  return it->second;
}

void StencilInstantiation::mapExprToAccessID(const std::shared_ptr<Expr>& expr, int accessID) {
  metadata_.ExprIDToAccessIDMap_.emplace(expr->getID(), accessID);
}

void StencilInstantiation::eraseExprToAccessID(std::shared_ptr<Expr> expr) {
  DAWN_ASSERT(metadata_.ExprIDToAccessIDMap_.count(expr->getID()));
  metadata_.ExprIDToAccessIDMap_.erase(expr->getID());
}

void StencilInstantiation::mapStmtToAccessID(const std::shared_ptr<Stmt>& stmt, int accessID) {
  metadata_.StmtIDToAccessIDMap_.emplace(stmt->getID(), accessID);
}

const std::string& StencilInstantiation::getNameFromLiteralAccessID(int AccessID) const {
  DAWN_ASSERT_MSG(isLiteral(AccessID), "Invalid literal");
  return metadata_.LiteralAccessIDToNameMap_.find(AccessID)->second;
}

bool StencilInstantiation::isGlobalVariable(const std::string& name) const {
  auto it = getNameToAccessIDMap().find(name);
  return it == getNameToAccessIDMap().end() ? false : isGlobalVariable(it->second);
}

void StencilInstantiation::insertStencilFunctionIntoSIR(
    const std::shared_ptr<sir::StencilFunction>& sirStencilFunction) {

  metadata_.allStencilFunctions_.push_back(sirStencilFunction);
}

bool StencilInstantiation::insertBoundaryConditions(std::string originalFieldName,
                                                    std::shared_ptr<BoundaryConditionDeclStmt> bc) {
  if(metadata_.FieldnameToBoundaryConditionMap_.count(originalFieldName) != 0) {
    return false;
  } else {
    metadata_.FieldnameToBoundaryConditionMap_.emplace(originalFieldName, bc);
    return true;
  }
}

Array3i StencilInstantiation::getFieldDimensionsMask(int FieldID) const {
  if(metadata_.fieldIDToInitializedDimensionsMap_.count(FieldID) == 0) {
    return Array3i{{1, 1, 1}};
  }
  return metadata_.fieldIDToInitializedDimensionsMap_.find(FieldID)->second;
}
const sir::Value& StencilInstantiation::getGlobalVariableValue(const std::string& name) const {
  auto it = metadata_.globalVariableMap_.find(name);
  DAWN_ASSERT(it != metadata_.globalVariableMap_.end());
  return *it->second;
}

ArrayRef<int> StencilInstantiation::getFieldVersions(int AccessID) const {
  return metadata_.variableVersions_.hasVariableMultipleVersions(AccessID)
             ? ArrayRef<int>(*(metadata_.variableVersions_.getVersions(AccessID)))
             : ArrayRef<int>{};
}

int StencilInstantiation::createVersionAndRename(int AccessID, Stencil* stencil, int curStageIdx,
                                                 int curStmtIdx, std::shared_ptr<Expr>& expr,
                                                 RenameDirection dir) {
  int newAccessID = nextUID();

  if(isField(AccessID)) {
    if(metadata_.variableVersions_.hasVariableMultipleVersions(AccessID)) {
      // Field is already multi-versioned, append a new version
      auto versions = metadata_.variableVersions_.getVersions(AccessID);

      // Set the second to last field to be a temporary (only the first and the last field will be
      // real storages, all other versions will be temporaries)
      int lastAccessID = versions->back();
      metadata_.TemporaryFieldAccessIDSet_.insert(lastAccessID);
      IIR_->getAllocatedFieldAccessIDSet().erase(lastAccessID);

      // The field with version 0 contains the original name
      const std::string& originalName = getNameFromAccessID(versions->front());

      // Register the new field
      setAccessIDNamePairOfField(newAccessID, originalName + "_" + std::to_string(versions->size()),
                                 false);
      IIR_->getAllocatedFieldAccessIDSet().insert(newAccessID);

      versions->push_back(newAccessID);
      metadata_.variableVersions_.insert(newAccessID, versions);

    } else {
      const std::string& originalName = getNameFromAccessID(AccessID);

      // Register the new *and* old field as being multi-versioned and indicate code-gen it has to
      // allocate the second version
      auto versionsVecPtr = std::make_shared<std::vector<int>>();
      *versionsVecPtr = {AccessID, newAccessID};

      setAccessIDNamePairOfField(newAccessID, originalName + "_1", false);
      IIR_->getAllocatedFieldAccessIDSet().insert(newAccessID);

      metadata_.variableVersions_.insert(AccessID, versionsVecPtr);
      metadata_.variableVersions_.insert(newAccessID, versionsVecPtr);
    }
  } else {
    if(metadata_.variableVersions_.hasVariableMultipleVersions(AccessID)) {
      // Variable is already multi-versioned, append a new version
      auto versions = metadata_.variableVersions_.getVersions(AccessID);

      // The variable with version 0 contains the original name
      const std::string& originalName = getNameFromAccessID(versions->front());

      // Register the new variable
      setAccessIDNamePair(newAccessID, originalName + "_" + std::to_string(versions->size()));
      versions->push_back(newAccessID);
      metadata_.variableVersions_.insert(newAccessID, versions);

    } else {
      const std::string& originalName = getNameFromAccessID(AccessID);

      // Register the new *and* old variable as being multi-versioned
      auto versionsVecPtr = std::make_shared<std::vector<int>>();
      *versionsVecPtr = {AccessID, newAccessID};

      setAccessIDNamePair(newAccessID, originalName + "_1");
      metadata_.variableVersions_.insert(AccessID, versionsVecPtr);
      metadata_.variableVersions_.insert(newAccessID, versionsVecPtr);
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
        renameAccessIDInStmts(this, AccessID, newAccessID, doMethod.getChildren()[i]);
        renameAccessIDInAccesses(this, AccessID, newAccessID, doMethod.getChildren()[i]);
      }

    } else {
      renameAccessIDInStmts(this, AccessID, newAccessID, doMethod.getChildren());
      renameAccessIDInAccesses(this, AccessID, newAccessID, doMethod.getChildren());
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
  removeAccessID(oldAccessID);
}

void StencilInstantiation::promoteLocalVariableToTemporaryField(Stencil* stencil, int AccessID,
                                                                const Stencil::Lifetime& lifetime) {
  std::string varname = getNameFromAccessID(AccessID);
  std::string fieldname = StencilInstantiation::makeTemporaryFieldname(
      StencilInstantiation::extractLocalVariablename(varname), AccessID);

  // Replace all variable accesses with field accesses
  stencil->forEachStatementAccessesPair(
      [&](ArrayRef<std::unique_ptr<StatementAccessesPair>> statementAccessesPair) -> void {
        replaceVarWithFieldAccessInStmts(stencil, AccessID, fieldname, statementAccessesPair);
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
  DAWN_ASSERT_MSG(varDeclStmt,
                  "first access to variable (i.e lifetime.Begin) is not an `VarDeclStmt`");
  DAWN_ASSERT_MSG(!varDeclStmt->isArray(), "cannot promote local array to temporary field");

  auto fieldAccessExpr = std::make_shared<FieldAccessExpr>(fieldname);
  metadata_.ExprIDToAccessIDMap_.emplace(fieldAccessExpr->getID(), AccessID);
  auto assignmentExpr =
      std::make_shared<AssignmentExpr>(fieldAccessExpr, varDeclStmt->getInitList().front());
  auto exprStmt = std::make_shared<ExprStmt>(assignmentExpr);

  // Replace the statement
  statementAccessesPairs[lifetime.Begin.StatementIndex]->setStatement(
      std::make_shared<Statement>(exprStmt, oldStatement->StackTrace));

  // Remove the variable
  removeAccessID(AccessID);
  metadata_.StmtIDToAccessIDMap_.erase(oldStatement->ASTStmt->getID());

  // Register the field
  setAccessIDNamePairOfField(AccessID, fieldname, true);

  // Update the fields of the stages we modified
  stencil->updateFields(lifetime);
}

void StencilInstantiation::promoteTemporaryFieldToAllocatedField(int AccessID) {
  DAWN_ASSERT(isTemporaryField(AccessID));
  metadata_.TemporaryFieldAccessIDSet_.erase(AccessID);
  IIR_->getAllocatedFieldAccessIDSet().insert(AccessID);
}

void StencilInstantiation::demoteTemporaryFieldToLocalVariable(Stencil* stencil, int AccessID,
                                                               const Stencil::Lifetime& lifetime) {
  std::string fieldname = getNameFromAccessID(AccessID);
  std::string varname = StencilInstantiation::makeLocalVariablename(
      StencilInstantiation::extractTemporaryFieldname(fieldname), AccessID);

  // Replace all field accesses with variable accesses
  stencil->forEachStatementAccessesPair(
      [&](ArrayRef<std::unique_ptr<StatementAccessesPair>> statementAccessesPairs) -> void {
        replaceFieldWithVarAccessInStmts(stencil, AccessID, varname, statementAccessesPairs);
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
  removeAccessID(AccessID);

  // Register the variable
  setAccessIDNamePair(AccessID, varname);
  metadata_.StmtIDToAccessIDMap_.emplace(varDeclStmt->getID(), AccessID);

  // Update the fields of the stages we modified
  stencil->updateFields(lifetime);
}

int StencilInstantiation::getAccessIDFromName(const std::string& name) const {
  auto it = getNameToAccessIDMap().find(name);
  DAWN_ASSERT_MSG(it != getNameToAccessIDMap().end(), "Invalid name");
  return it->second;
}

int StencilInstantiation::getAccessIDFromExpr(const std::shared_ptr<Expr>& expr) const {
    auto it = metadata_.ExprIDToAccessIDMap_.find(expr->getID());
    DAWN_ASSERT_MSG(it != metadata_.ExprIDToAccessIDMap_.end(), "Invalid Expr");
    return it->second;
}

int StencilInstantiation::getAccessIDFromStmt(const std::shared_ptr<Stmt>& stmt) const {
    auto it = metadata_.StmtIDToAccessIDMap_.find(stmt->getID());
    DAWN_ASSERT_MSG(it != metadata_.StmtIDToAccessIDMap_.end(), "Invalid Stmt");
    return it->second;
}

void StencilInstantiation::setAccessIDOfStmt(const std::shared_ptr<Stmt>& stmt,
                                             const int accessID) {
  DAWN_ASSERT(metadata_.StmtIDToAccessIDMap_.count(stmt->getID()));
  metadata_.StmtIDToAccessIDMap_[stmt->getID()] = accessID;
}

void StencilInstantiation::setAccessIDOfExpr(const std::shared_ptr<Expr>& expr,
                                             const int accessID) {
  DAWN_ASSERT(metadata_.ExprIDToAccessIDMap_.count(expr->getID()));
  metadata_.ExprIDToAccessIDMap_[expr->getID()] = accessID;
}

void StencilInstantiation::removeStencilFunctionInstantiation(
    const std::shared_ptr<StencilFunCallExpr>& expr,
    std::shared_ptr<StencilFunctionInstantiation> callerStencilFunctionInstantiation) {

  std::shared_ptr<StencilFunctionInstantiation> func = nullptr;

  if(callerStencilFunctionInstantiation) {
    func = callerStencilFunctionInstantiation->getStencilFunctionInstantiation(expr);
    callerStencilFunctionInstantiation->removeStencilFunctionInstantiation(expr);
  } else {
    func = getStencilFunctionInstantiation(expr);
    metadata_.ExprToStencilFunctionInstantiationMap_.erase(expr);
  }

  for(auto it = metadata_.stencilFunctionInstantiations_.begin();
      it != metadata_.stencilFunctionInstantiations_.end();) {
    if(*it == func)
      it = metadata_.stencilFunctionInstantiations_.erase(it);
    else
      ++it;
  }
}

const std::shared_ptr<StencilFunctionInstantiation>
StencilInstantiation::getStencilFunctionInstantiation(
    const std::shared_ptr<StencilFunCallExpr>& expr) const {
  auto it = metadata_.ExprToStencilFunctionInstantiationMap_.find(expr);
  DAWN_ASSERT_MSG(it != metadata_.ExprToStencilFunctionInstantiationMap_.end(),
                  "Invalid stencil function");
  return it->second;
}

std::shared_ptr<StencilFunctionInstantiation>
StencilInstantiation::getStencilFunctionInstantiationCandidate(
    const std::shared_ptr<StencilFunCallExpr>& expr) {
  auto it = std::find_if(
      metadata_.stencilFunInstantiationCandidate_.begin(),
      metadata_.stencilFunInstantiationCandidate_.end(),
      [&](std::pair<std::shared_ptr<StencilFunctionInstantiation>,
                    StencilMetaInformation::StencilFunctionInstantiationCandidate> const& pair) {
        return (pair.first->getExpression() == expr);
      });
  DAWN_ASSERT_MSG((it != metadata_.stencilFunInstantiationCandidate_.end()),
                  "stencil function candidate not found");

  return it->first;
}

std::shared_ptr<StencilFunctionInstantiation>
StencilInstantiation::getStencilFunctionInstantiationCandidate(const std::string stencilFunName) {
  auto it = std::find_if(
      metadata_.stencilFunInstantiationCandidate_.begin(),
      metadata_.stencilFunInstantiationCandidate_.end(),
      [&](std::pair<std::shared_ptr<StencilFunctionInstantiation>,
                    StencilMetaInformation::StencilFunctionInstantiationCandidate> const& pair) {
        return (pair.first->getExpression()->getCallee() == stencilFunName);
      });
  DAWN_ASSERT_MSG((it != metadata_.stencilFunInstantiationCandidate_.end()),
                  "stencil function candidate not found");

  return it->first;
}

std::shared_ptr<StencilFunctionInstantiation> StencilInstantiation::cloneStencilFunctionCandidate(
    const std::shared_ptr<StencilFunctionInstantiation>& stencilFun, std::string functionName) {
  DAWN_ASSERT(metadata_.stencilFunInstantiationCandidate_.count(stencilFun));
  auto stencilFunClone = std::make_shared<StencilFunctionInstantiation>(stencilFun->clone());

  auto stencilFunExpr =
      std::dynamic_pointer_cast<StencilFunCallExpr>(stencilFun->getExpression()->clone());
  stencilFunExpr->setCallee(functionName);

  auto sirStencilFun = std::make_shared<sir::StencilFunction>(*(stencilFun->getStencilFunction()));
  sirStencilFun->Name = functionName;

  stencilFunClone->setExpression(stencilFunExpr);
  stencilFunClone->setStencilFunction(sirStencilFun);

  metadata_.stencilFunInstantiationCandidate_.emplace(
      stencilFunClone, metadata_.stencilFunInstantiationCandidate_[stencilFun]);
  return stencilFunClone;
}

const std::unordered_map<std::shared_ptr<StencilFunCallExpr>,
                         std::shared_ptr<StencilFunctionInstantiation>>&
StencilInstantiation::getExprToStencilFunctionInstantiationMap() const {
  return metadata_.ExprToStencilFunctionInstantiationMap_;
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

  metadata_.stencilFunInstantiationCandidate_.emplace(
      stencilFun, StencilMetaInformation::StencilFunctionInstantiationCandidate{
                      curStencilFunctionInstantiation});

  return stencilFun;
}

void StencilInstantiation::insertExprToStencilFunction(
    std::shared_ptr<StencilFunctionInstantiation> stencilFun) {
  metadata_.ExprToStencilFunctionInstantiationMap_.emplace(stencilFun->getExpression(), stencilFun);
}

void StencilInstantiation::deregisterStencilFunction(
    std::shared_ptr<StencilFunctionInstantiation> stencilFun) {

  bool found = RemoveIf(metadata_.ExprToStencilFunctionInstantiationMap_,
                        [&](std::pair<std::shared_ptr<StencilFunCallExpr>,
                                      std::shared_ptr<StencilFunctionInstantiation>>
                                pair) { return (pair.second == stencilFun); });
  // make sure the element existed and was removed
  DAWN_ASSERT(found);
  found = RemoveIf(
      metadata_.stencilFunctionInstantiations_,
      [&](const std::shared_ptr<StencilFunctionInstantiation>& v) { return (v == stencilFun); });

  // make sure the element existed and was removed
  DAWN_ASSERT(found);
}

void StencilInstantiation::finalizeStencilFunctionSetup(
    std::shared_ptr<StencilFunctionInstantiation> stencilFun) {

  DAWN_ASSERT(metadata_.stencilFunInstantiationCandidate_.count(stencilFun));
  stencilFun->closeFunctionBindings();
  // We take the candidate to stencil function and placed it in the stencil function instantiations
  // container
  StencilMetaInformation::StencilFunctionInstantiationCandidate candidate =
      metadata_.stencilFunInstantiationCandidate_[stencilFun];

  // map of expr to stencil function instantiation is updated
  if(candidate.callerStencilFunction_) {
    candidate.callerStencilFunction_->insertExprToStencilFunction(stencilFun);
  } else {
    insertExprToStencilFunction(stencilFun);
  }

  stencilFun->update();

  metadata_.stencilFunctionInstantiations_.push_back(stencilFun);
  // we remove the candidate to stencil function
  metadata_.stencilFunInstantiationCandidate_.erase(stencilFun);
}

std::unordered_map<std::shared_ptr<StencilCallDeclStmt>, int>&
StencilInstantiation::getStencilCallToStencilIDMap() {
  return IIR_->getStencilCallToStencilIDMap();
}

const std::unordered_map<std::shared_ptr<StencilCallDeclStmt>, int>&
StencilInstantiation::getStencilCallToStencilIDMap() const {
  return IIR_->getStencilCallToStencilIDMap();
}

std::unordered_map<int, std::shared_ptr<StencilCallDeclStmt>>&
StencilInstantiation::getIDToStencilCallMap() {
  return metadata_.IDToStencilCallMap_;
}

const std::unordered_map<int, std::shared_ptr<StencilCallDeclStmt>>&
StencilInstantiation::getIDToStencilCallMap() const {
  return metadata_.IDToStencilCallMap_;
}

int StencilInstantiation::getStencilIDFromStmt(
    const std::shared_ptr<StencilCallDeclStmt>& stmt) const {
  auto it = IIR_->getStencilCallToStencilIDMap().find(stmt);
  DAWN_ASSERT_MSG(it != IIR_->getStencilCallToStencilIDMap().end(), "Invalid stencil call");
  return it->second;
}

std::unordered_map<std::string, int>& StencilInstantiation::getNameToAccessIDMap() {
  return IIR_->getNameToAccessIDs();
}

const std::unordered_map<std::string, int>& StencilInstantiation::getNameToAccessIDMap() const {
  return IIR_->getNameToAccessIDs();
}

std::unordered_map<int, std::string>& StencilInstantiation::getAccessIDToNameMap() {
  return metadata_.AccessIDToNameMap_;
}

const std::unordered_map<int, std::string>& StencilInstantiation::getAccessIDToNameMap() const {
  return metadata_.AccessIDToNameMap_;
}

std::unordered_map<int, std::string>& StencilInstantiation::getLiteralAccessIDToNameMap() {
  return metadata_.LiteralAccessIDToNameMap_;
}
const std::unordered_map<int, std::string>&
StencilInstantiation::getLiteralAccessIDToNameMap() const {
  return metadata_.LiteralAccessIDToNameMap_;
}

std::set<int>& StencilInstantiation::getFieldAccessIDSet() { return metadata_.FieldAccessIDSet_; }

const std::set<int>& StencilInstantiation::getFieldAccessIDSet() const {
  return metadata_.FieldAccessIDSet_;
}

std::set<int>& StencilInstantiation::getGlobalVariableAccessIDSet() {
  return metadata_.GlobalVariableAccessIDSet_;
}

const std::set<int>& StencilInstantiation::getGlobalVariableAccessIDSet() const {
  return metadata_.GlobalVariableAccessIDSet_;
}

namespace {

/// @brief Get the orignal name of the field (or variable) given by AccessID and a list of
/// SourceLocations where this field (or variable) was accessed.
class OriginalNameGetter : public ASTVisitorForwarding {
  const StencilInstantiation* instantiation_;
  const int AccessID_;
  const bool captureLocation_;

  std::string name_;
  std::vector<SourceLocation> locations_;

public:
  OriginalNameGetter(const StencilInstantiation* instantiation, int AccessID, bool captureLocation)
      : instantiation_(instantiation), AccessID_(AccessID), captureLocation_(captureLocation) {}

  virtual void visit(const std::shared_ptr<VarDeclStmt>& stmt) override {
    if(instantiation_->getAccessIDFromStmt(stmt) == AccessID_) {
      name_ = stmt->getName();
      if(captureLocation_)
        locations_.push_back(stmt->getSourceLocation());
    }

    for(const auto& expr : stmt->getInitList())
      expr->accept(*this);
  }

  void visit(const std::shared_ptr<VarAccessExpr>& expr) override {
    if(instantiation_->getAccessIDFromExpr(expr) == AccessID_) {
      name_ = expr->getName();
      if(captureLocation_)
        locations_.push_back(expr->getSourceLocation());
    }
  }

  void visit(const std::shared_ptr<LiteralAccessExpr>& expr) override {
    if(instantiation_->getAccessIDFromExpr(expr) == AccessID_) {
      name_ = expr->getValue();
      if(captureLocation_)
        locations_.push_back(expr->getSourceLocation());
    }
  }

  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) override {
    if(instantiation_->getAccessIDFromExpr(expr) == AccessID_) {
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
  OriginalNameGetter orignalNameGetter(this, AccessID, true);
  stmt->accept(orignalNameGetter);
  return orignalNameGetter.getNameLocationPair();
}

std::string StencilInstantiation::getOriginalNameFromAccessID(int AccessID) const {
  OriginalNameGetter orignalNameGetter(this, AccessID, true);

  for(const auto& stmtAccessesPair : iterateIIROver<StatementAccessesPair>(*getIIR())) {
    stmtAccessesPair->getStatement()->ASTStmt->accept(orignalNameGetter);
    if(orignalNameGetter.hasName())
      return orignalNameGetter.getName();
  }

  // Best we can do...
  return getNameFromAccessID(AccessID);
}

namespace {

template <int Level>
struct PrintDescLine {
  PrintDescLine(const Twine& name) {
    std::cout << MakeIndent<Level>::value << format("\e[1;3%im", Level) << name.str() << "\n"
              << MakeIndent<Level>::value << "{\n\e[0m";
  }
  ~PrintDescLine() { std::cout << MakeIndent<Level>::value << format("\e[1;3%im}\n\e[0m", Level); }
};

} // anonymous namespace

bool StencilInstantiation::checkTreeConsistency() const { return IIR_->checkTreeConsistency(); }

void StencilInstantiation::dump() const {
  std::cout << "StencilInstantiation : " << getName() << "\n";

  int i = 0;
  for(const auto& stencil : getStencils()) {
    PrintDescLine<1> iline("Stencil_" + Twine(i));

    int j = 0;
    const auto& multiStages = stencil->getChildren();
    for(const auto& multiStage : multiStages) {
      PrintDescLine<2> jline(Twine("MultiStage_") + Twine(j) + " [" +
                             loopOrderToString(multiStage->getLoopOrder()) + "]");

      int k = 0;
      const auto& stages = multiStage->getChildren();
      for(const auto& stage : stages) {
        PrintDescLine<3> kline(Twine("Stage_") + Twine(k));

        int l = 0;
        const auto& doMethods = stage->getChildren();
        for(const auto& doMethod : doMethods) {
          PrintDescLine<4> lline(Twine("Do_") + Twine(l) + " " +
                                 doMethod->getInterval().toString());

          const auto& statementAccessesPairs = doMethod->getChildren();
          for(std::size_t m = 0; m < statementAccessesPairs.size(); ++m) {
            std::cout << "\e[1m"
                      << ASTStringifer::toString(statementAccessesPairs[m]->getStatement()->ASTStmt,
                                                 5 * DAWN_PRINT_INDENT)
                      << "\e[0m";
            std::cout << statementAccessesPairs[m]->getAccesses()->toString(this,
                                                                            6 * DAWN_PRINT_INDENT)
                      << "\n";
          }
          l += 1;
        }
        std::cout << "\e[1m" << std::string(4 * DAWN_PRINT_INDENT, ' ')
                  << "Extents: " << stage->getExtents() << std::endl
                  << "\e[0m";
        k += 1;
      }
      j += 1;
    }
    ++i;
  }
  std::cout.flush();
}

void StencilInstantiation::dumpAsJson(std::string filename, std::string passName) const {
  json::json jout;

  int i = 0;
  for(const auto& stencil : getStencils()) {
    json::json jStencil;

    int j = 0;
    for(const auto& multiStage : stencil->getChildren()) {
      json::json jMultiStage;
      jMultiStage["LoopOrder"] = loopOrderToString(multiStage->getLoopOrder());

      int k = 0;
      const auto& stages = multiStage->getChildren();
      for(const auto& stage : stages) {
        json::json jStage;

        int l = 0;
        for(const auto& doMethod : stage->getChildren()) {
          json::json jDoMethod;

          jDoMethod["Interval"] = doMethod->getInterval().toString();

          const auto& statementAccessesPairs = doMethod->getChildren();
          for(std::size_t m = 0; m < statementAccessesPairs.size(); ++m) {
            jDoMethod["Stmt_" + std::to_string(m)] = ASTStringifer::toString(
                statementAccessesPairs[m]->getStatement()->ASTStmt, 0, false);
            jDoMethod["Accesses_" + std::to_string(m)] =
                statementAccessesPairs[m]->getAccesses()->reportAccesses(this);
          }

          jStage["Do_" + std::to_string(l++)] = jDoMethod;
        }

        jMultiStage["Stage_" + std::to_string(k++)] = jStage;
      }

      jStencil["MultiStage_" + std::to_string(j++)] = jMultiStage;
    }

    if(passName.empty())
      jout[getName()]["Stencil_" + std::to_string(i)] = jStencil;
    else
      jout[passName][getName()]["Stencil_" + std::to_string(i)] = jStencil;
    ++i;
  }

  std::ofstream fs(filename, std::ios::out | std::ios::trunc);
  if(!fs.is_open()) {
    DiagnosticsBuilder diag(DiagnosticsKind::Error, SourceLocation());
    diag << "file system error: cannot open file: " << filename;
    context_->getDiagnostics().report(diag);
  }

  fs << jout.dump(2) << std::endl;
  fs.close();
}

static std::string makeNameImpl(const char* prefix, const std::string& name, int AccessID) {
  return prefix + name + "_" + std::to_string(AccessID);
}

static std::string extractNameImpl(StringRef prefix, const std::string& name) {
  StringRef nameRef(name);

  // Remove leading `prefix`
  std::size_t leadingLocalPos = nameRef.find(prefix);
  nameRef = nameRef.drop_front(leadingLocalPos != StringRef::npos ? prefix.size() : 0);

  // Remove trailing `_X` where X is the AccessID
  std::size_t trailingAccessIDPos = nameRef.find_last_of('_');
  nameRef = nameRef.drop_back(
      trailingAccessIDPos != StringRef::npos ? nameRef.size() - trailingAccessIDPos : 0);

  return nameRef.empty() ? name : nameRef.str();
}

std::string StencilInstantiation::makeLocalVariablename(const std::string& name, int AccessID) {
  return makeNameImpl("__local_", name, AccessID);
}

std::string StencilInstantiation::makeTemporaryFieldname(const std::string& name, int AccessID) {
  return makeNameImpl("__tmp_", name, AccessID);
}

std::string StencilInstantiation::extractLocalVariablename(const std::string& name) {
  return extractNameImpl("__local_", name);
}

std::string StencilInstantiation::extractTemporaryFieldname(const std::string& name) {
  return extractNameImpl("__tmp_", name);
}

std::string StencilInstantiation::makeStencilCallCodeGenName(int StencilID) {
  return "__code_gen_" + std::to_string(StencilID);
}

bool StencilInstantiation::isStencilCallCodeGenName(const std::string& name) {
  return StringRef(name).startswith("__code_gen_");
}

void StencilInstantiation::reportAccesses() const {
  // Stencil functions
  for(const auto& stencilFun : metadata_.stencilFunctionInstantiations_) {
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
              << stmtAccessesPair->getAccesses()->reportAccesses(this) << "\n";
  }
}

} // namespace iir
} // namespace dawn
