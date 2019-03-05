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
    : context_(context), IIR_(make_unique<IIR>()), SIR_(context->getSIR()) {}

StencilMetaInformation& StencilInstantiation::getMetaData() { return metadata_; }

std::shared_ptr<StencilInstantiation> StencilInstantiation::clone() const {

  std::shared_ptr<StencilInstantiation> stencilInstantiation =
      std::make_shared<StencilInstantiation>(context_);

  stencilInstantiation->metadata_.clone(metadata_);

  stencilInstantiation->IIR_ = make_unique<iir::IIR>();
  IIR_->clone(stencilInstantiation->IIR_);

  return stencilInstantiation;
}

const std::string StencilInstantiation::getName() const { return metadata_.stencilName_; }

void StencilInstantiation::insertStencilFunctionIntoSIR(
    const std::shared_ptr<sir::StencilFunction>& sirStencilFunction) {
  SIR_->StencilFunctions.push_back(sirStencilFunction);
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

  if(metadata_.isField(AccessID)) {
    if(metadata_.variableVersions_.hasVariableMultipleVersions(AccessID)) {
      // Field is already multi-versioned, append a new version
      auto versions = metadata_.variableVersions_.getVersions(AccessID);

      // Set the second to last field to be a temporary (only the first and the last field will be
      // real storages, all other versions will be temporaries)
      int lastAccessID = versions->back();
      metadata_.TemporaryFieldAccessIDSet_.insert(lastAccessID);
      metadata_.eraseAllocatedField(lastAccessID);

      // The field with version 0 contains the original name
      const std::string& originalName = metadata_.getFieldNameFromAccessID(versions->front());

      // Register the new field
      metadata_.setAccessIDNamePairOfField(
          newAccessID, originalName + "_" + std::to_string(versions->size()), false);
      metadata_.insertAllocatedField(newAccessID);

      versions->push_back(newAccessID);
      metadata_.variableVersions_.insert(newAccessID, versions);

    } else {
      const std::string& originalName = metadata_.getFieldNameFromAccessID(AccessID);

      // Register the new *and* old field as being multi-versioned and indicate code-gen it has to
      // allocate the second version
      auto versionsVecPtr = std::make_shared<std::vector<int>>();
      *versionsVecPtr = {AccessID, newAccessID};

      metadata_.setAccessIDNamePairOfField(newAccessID, originalName + "_1", false);
      metadata_.insertAllocatedField(newAccessID);

      metadata_.variableVersions_.insert(AccessID, versionsVecPtr);
      metadata_.variableVersions_.insert(newAccessID, versionsVecPtr);
    }
  } else {
    if(metadata_.variableVersions_.hasVariableMultipleVersions(AccessID)) {
      // Variable is already multi-versioned, append a new version
      auto versions = metadata_.variableVersions_.getVersions(AccessID);

      // The variable with version 0 contains the original name
      const std::string& originalName = metadata_.getFieldNameFromAccessID(versions->front());

      // Register the new variable
      metadata_.setAccessIDNamePair(newAccessID,
                                    originalName + "_" + std::to_string(versions->size()));
      versions->push_back(newAccessID);
      metadata_.variableVersions_.insert(newAccessID, versions);

    } else {
      const std::string& originalName = metadata_.getFieldNameFromAccessID(AccessID);

      // Register the new *and* old variable as being multi-versioned
      auto versionsVecPtr = std::make_shared<std::vector<int>>();
      *versionsVecPtr = {AccessID, newAccessID};

      metadata_.setAccessIDNamePair(newAccessID, originalName + "_1");
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
  std::string fieldname = StencilInstantiation::makeTemporaryFieldname(
      StencilInstantiation::extractLocalVariablename(varname), accessID);

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
  if(!varDeclStmt && temporaryScope != TemporaryScope::TT_Field) {
    throw std::runtime_error(format("Promote local variable to temporary field: a var decl is not "
                                    "found for accessid: %i , name :%s",
                                    accessID, metadata_.getNameFromAccessID(accessID)));
  }
  if(varDeclStmt) {
    DAWN_ASSERT_MSG(!varDeclStmt->isArray(), "cannot promote local array to temporary field");

    auto fieldAccessExpr = std::make_shared<FieldAccessExpr>(fieldname);
    metadata_.ExprIDToAccessIDMap_.emplace(fieldAccessExpr->getID(), accessID);
    auto assignmentExpr =
        std::make_shared<AssignmentExpr>(fieldAccessExpr, varDeclStmt->getInitList().front());
    auto exprStmt = std::make_shared<ExprStmt>(assignmentExpr);

    // Replace the statement
    statementAccessesPairs[lifetime.Begin.StatementIndex]->setStatement(
        std::make_shared<Statement>(exprStmt, oldStatement->StackTrace));

    // Remove the variable
    metadata_.removeAccessID(accessID);
    metadata_.StmtIDToAccessIDMap_.erase(oldStatement->ASTStmt->getID());
  }
  // Register the field
  metadata_.setAccessIDNamePairOfField(accessID, fieldname, true);

  // Update the fields of the stages we modified
  stencil->updateFields(lifetime);
}

void StencilInstantiation::promoteTemporaryFieldToAllocatedField(int AccessID) {
  DAWN_ASSERT(metadata_.isTemporaryField(AccessID));
  metadata_.TemporaryFieldAccessIDSet_.erase(AccessID);
  metadata_.insertAllocatedField(AccessID);
}

void StencilInstantiation::demoteTemporaryFieldToLocalVariable(Stencil* stencil, int AccessID,
                                                               const Stencil::Lifetime& lifetime) {
  std::string fieldname = metadata_.getFieldNameFromAccessID(AccessID);
  std::string varname = StencilInstantiation::makeLocalVariablename(
      StencilInstantiation::extractTemporaryFieldname(fieldname), AccessID);

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
  metadata_.StmtIDToAccessIDMap_.emplace(varDeclStmt->getID(), AccessID);

  // Update the fields of the stages we modified
  stencil->updateFields(lifetime);
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
  return metadata_.getStencilFunctionInstantiation(expr);
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
StencilInstantiation::getStencilFunctionInstantiationCandidate(const std::string stencilFunName,
                                                               const Interval& interval) {
  auto it = std::find_if(
      metadata_.stencilFunInstantiationCandidate_.begin(),
      metadata_.stencilFunInstantiationCandidate_.end(),
      [&](std::pair<std::shared_ptr<StencilFunctionInstantiation>,
                    StencilMetaInformation::StencilFunctionInstantiationCandidate> const& pair) {
        return (pair.first->getExpression()->getCallee() == stencilFunName &&
                (pair.first->getInterval() == interval));
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
              << stmtAccessesPair->getAccesses()->reportAccesses(metadata_) << "\n";
  }
}

} // namespace iir
} // namespace dawn
