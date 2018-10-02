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

#include "dawn/IIR/MetaInformation.h"
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

namespace{
/// @brief Get the orignal name of the field (or variable) given by AccessID and a list of
/// SourceLocations where this field (or variable) was accessed.
class OriginalNameGetter : public ASTVisitorForwarding {
  const StencilMetaInformation* stencilInfo_;
  const int AccessID_;
  const bool captureLocation_;

  std::string name_;
  std::vector<SourceLocation> locations_;

public:
  OriginalNameGetter(const StencilMetaInformation* stencilInfo, int AccessID, bool captureLocation)
      : stencilInfo_(stencilInfo), AccessID_(AccessID), captureLocation_(captureLocation) {}

  virtual void visit(const std::shared_ptr<VarDeclStmt>& stmt) override {
    if(stencilInfo_->getAccessIDFromStmt(stmt) == AccessID_) {
      name_ = stmt->getName();
      if(captureLocation_)
        locations_.push_back(stmt->getSourceLocation());
    }

    for(const auto& expr : stmt->getInitList())
      expr->accept(*this);
  }

  void visit(const std::shared_ptr<VarAccessExpr>& expr) override {
    if(stencilInfo_->getAccessIDFromExpr(expr) == AccessID_) {
      name_ = expr->getName();
      if(captureLocation_)
        locations_.push_back(expr->getSourceLocation());
    }
  }

  void visit(const std::shared_ptr<LiteralAccessExpr>& expr) override {
    if(stencilInfo_->getAccessIDFromExpr(expr) == AccessID_) {
      name_ = expr->getValue();
      if(captureLocation_)
        locations_.push_back(expr->getSourceLocation());
    }
  }

  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) override {
    if(stencilInfo_->getAccessIDFromExpr(expr) == AccessID_) {
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
}  // anonymous namespace

//===------------------------------------------------------------------------------------------===//
//     StencilInstantiation
//===------------------------------------------------------------------------------------------===//

StencilMetaInformation::StencilMetaInformation() {}

void StencilMetaInformation::setAccessIDNamePair(int AccessID, const std::string& name) {
  AccessIDToNameMap_.emplace(AccessID, name);
  NameToAccessIDMap_.emplace(name, AccessID);
}

void StencilMetaInformation::setAccessIDNamePairOfField(int AccessID, const std::string& name,
                                                        bool isTemporary) {
  setAccessIDNamePair(AccessID, name);
  FieldAccessIDSet_.insert(AccessID);
  if(isTemporary)
    TemporaryFieldAccessIDSet_.insert(AccessID);
}

void StencilMetaInformation::setAccessIDNamePairOfGlobalVariable(int AccessID,
                                                                 const std::string& name) {
  setAccessIDNamePair(AccessID, name);
  GlobalVariableAccessIDSet_.insert(AccessID);
}

void StencilMetaInformation::removeAccessID(int AccessID) {
  if(NameToAccessIDMap_.count(AccessIDToNameMap_[AccessID]))
    NameToAccessIDMap_.erase(AccessIDToNameMap_[AccessID]);

  AccessIDToNameMap_.erase(AccessID);
  FieldAccessIDSet_.erase(AccessID);
  TemporaryFieldAccessIDSet_.erase(AccessID);

  if(variableVersions_.hasVariableMultipleVersions(AccessID)) {
    auto versions = variableVersions_.getVersions(AccessID);
    versions->erase(std::remove_if(versions->begin(), versions->end(),
                                   [&](int AID) { return AID == AccessID; }),
                    versions->end());
  }
}

const std::unordered_map<std::shared_ptr<Stmt>, int>&
StencilMetaInformation::getStmtToAccessIDMap() const {
  return StmtToAccessIDMap_;
}

std::unordered_map<std::shared_ptr<Stmt>, int>& StencilMetaInformation::getStmtToAccessIDMap() {
  return StmtToAccessIDMap_;
}

const std::string& StencilMetaInformation::getNameFromAccessID(int AccessID) const {
  if(AccessID < 0)
    return getNameFromLiteralAccessID(AccessID);
  auto it = AccessIDToNameMap_.find(AccessID);
  DAWN_ASSERT_MSG(it != AccessIDToNameMap_.end(), "Invalid AccessID");
  return it->second;
}

const std::string& StencilMetaInformation::getNameFromStageID(int StageID) const {
  auto it = StageIDToNameMap_.find(StageID);
  DAWN_ASSERT_MSG(it != StageIDToNameMap_.end(), "Invalid StageID");
  return it->second;
}

void StencilMetaInformation::mapExprToAccessID(const std::shared_ptr<Expr>& expr, int accessID) {
  ExprToAccessIDMap_.emplace(expr, accessID);
}

void StencilMetaInformation::eraseExprToAccessID(std::shared_ptr<Expr> expr) {
  DAWN_ASSERT(ExprToAccessIDMap_.count(expr));
  ExprToAccessIDMap_.erase(expr);
}

void StencilMetaInformation::mapStmtToAccessID(const std::shared_ptr<Stmt>& stmt, int accessID) {
  StmtToAccessIDMap_.emplace(stmt, accessID);
}

const std::string& StencilMetaInformation::getNameFromLiteralAccessID(int AccessID) const {
  DAWN_ASSERT_MSG(isLiteral(AccessID), "Invalid literal");
  return LiteralAccessIDToNameMap_.find(AccessID)->second;
}

bool StencilMetaInformation::isGlobalVariable(const std::string& name) const {
  auto it = NameToAccessIDMap_.find(name);
  return it == NameToAccessIDMap_.end() ? false : isGlobalVariable(it->second);
}

// void StencilMetaInformation::insertStencilFunctionIntoSIR(
//    const std::shared_ptr<sir::StencilFunction>& sirStencilFunction) {
//  SIR_->StencilFunctions.push_back(sirStencilFunction);
//}

bool StencilMetaInformation::insertBoundaryConditions(
    std::string originalFieldName, std::shared_ptr<BoundaryConditionDeclStmt> bc) {
  if(FieldnameToBoundaryConditionMap_.count(originalFieldName) != 0) {
    return false;
  } else {
    FieldnameToBoundaryConditionMap_.emplace(originalFieldName, bc);
    return true;
  }
}

Array3i StencilMetaInformation::getFieldDimensionsMask(int FieldID) {
  if(fieldIDToInitializedDimensionsMap_.count(FieldID) == 0) {
    return Array3i{{1, 1, 1}};
  }
  return fieldIDToInitializedDimensionsMap_.find(FieldID)->second;
}
const sir::Value& StencilMetaInformation::getGlobalVariableValue(const std::string& name) const {
  auto it = globalVariableMap_.find(name);
  DAWN_ASSERT(it != globalVariableMap_.end());
  return *it->second;
}

ArrayRef<int> StencilMetaInformation::getFieldVersions(int AccessID) const {
  return variableVersions_.hasVariableMultipleVersions(AccessID)
             ? ArrayRef<int>(*(variableVersions_.getVersions(AccessID)))
             : ArrayRef<int>{};
}

int StencilMetaInformation::getAccessIDFromName(const std::string& name) const {
  auto it = NameToAccessIDMap_.find(name);
  DAWN_ASSERT_MSG(it != NameToAccessIDMap_.end(), "Invalid name");
  return it->second;
}

int StencilMetaInformation::getAccessIDFromExpr(const std::shared_ptr<Expr>& expr) const {
  auto it = ExprToAccessIDMap_.find(expr);
  DAWN_ASSERT_MSG(it != ExprToAccessIDMap_.end(), "Invalid Expr");
  return it->second;
}

int StencilMetaInformation::getAccessIDFromStmt(const std::shared_ptr<Stmt>& stmt) const {
  auto it = StmtToAccessIDMap_.find(stmt);
  DAWN_ASSERT_MSG(it != StmtToAccessIDMap_.end(), "Invalid Stmt");
  return it->second;
}

void StencilMetaInformation::setAccessIDOfStmt(const std::shared_ptr<Stmt>& stmt,
                                               const int accessID) {
  DAWN_ASSERT(StmtToAccessIDMap_.count(stmt));
  StmtToAccessIDMap_[stmt] = accessID;
}

void StencilMetaInformation::setAccessIDOfExpr(const std::shared_ptr<Expr>& expr,
                                               const int accessID) {
  DAWN_ASSERT(ExprToAccessIDMap_.count(expr));
  ExprToAccessIDMap_[expr] = accessID;
}

const std::shared_ptr<StencilFunctionInstantiation>
StencilMetaInformation::getStencilFunctionInstantiation(
    const std::shared_ptr<StencilFunCallExpr>& expr) const {
  auto it = ExprToStencilFunctionInstantiationMap_.find(expr);
  DAWN_ASSERT_MSG(it != ExprToStencilFunctionInstantiationMap_.end(), "Invalid stencil function");
  return it->second;
}

std::shared_ptr<StencilFunctionInstantiation>
StencilMetaInformation::getStencilFunctionInstantiationCandidate(
    const std::shared_ptr<StencilFunCallExpr>& expr) {
  auto it = std::find_if(stencilFunInstantiationCandidate_.begin(),
                         stencilFunInstantiationCandidate_.end(),
                         [&](std::pair<std::shared_ptr<StencilFunctionInstantiation>,
                                       StencilFunctionInstantiationCandidate> const& pair) {
                           return (pair.first->getExpression() == expr);
                         });
  DAWN_ASSERT_MSG((it != stencilFunInstantiationCandidate_.end()),
                  "stencil function candidate not found");

  return it->first;
}

std::shared_ptr<StencilFunctionInstantiation>
StencilMetaInformation::getStencilFunctionInstantiationCandidate(const std::string stencilFunName) {
  auto it = std::find_if(stencilFunInstantiationCandidate_.begin(),
                         stencilFunInstantiationCandidate_.end(),
                         [&](std::pair<std::shared_ptr<StencilFunctionInstantiation>,
                                       StencilFunctionInstantiationCandidate> const& pair) {
                           return (pair.first->getExpression()->getCallee() == stencilFunName);
                         });
  DAWN_ASSERT_MSG((it != stencilFunInstantiationCandidate_.end()),
                  "stencil function candidate not found");

  return it->first;
}

std::shared_ptr<StencilFunctionInstantiation> StencilMetaInformation::cloneStencilFunctionCandidate(
    const std::shared_ptr<StencilFunctionInstantiation>& stencilFun, std::string functionName) {
  DAWN_ASSERT(stencilFunInstantiationCandidate_.count(stencilFun));
  auto stencilFunClone = std::make_shared<StencilFunctionInstantiation>(stencilFun->clone());

  auto stencilFunExpr =
      std::dynamic_pointer_cast<StencilFunCallExpr>(stencilFun->getExpression()->clone());
  stencilFunExpr->setCallee(functionName);

  auto sirStencilFun = std::make_shared<sir::StencilFunction>(*(stencilFun->getStencilFunction()));
  sirStencilFun->Name = functionName;

  stencilFunClone->setExpression(stencilFunExpr);
  stencilFunClone->setStencilFunction(sirStencilFun);

  stencilFunInstantiationCandidate_.emplace(stencilFunClone,
                                            stencilFunInstantiationCandidate_[stencilFun]);
  return stencilFunClone;
}

const std::unordered_map<std::shared_ptr<StencilFunCallExpr>,
                         std::shared_ptr<StencilFunctionInstantiation>>&
StencilMetaInformation::getExprToStencilFunctionInstantiationMap() const {
  return ExprToStencilFunctionInstantiationMap_;
}

std::unordered_map<std::shared_ptr<StencilCallDeclStmt>, int>&
StencilMetaInformation::getStencilCallToStencilIDMap() {
  return StencilCallToStencilIDMap_;
}

const std::unordered_map<std::shared_ptr<StencilCallDeclStmt>, int>&
StencilMetaInformation::getStencilCallToStencilIDMap() const {
  return StencilCallToStencilIDMap_;
}

std::unordered_map<int, std::shared_ptr<StencilCallDeclStmt>>&
StencilMetaInformation::getIDToStencilCallMap() {
  return IDToStencilCallMap_;
}

const std::unordered_map<int, std::shared_ptr<StencilCallDeclStmt>>&
StencilMetaInformation::getIDToStencilCallMap() const {
  return IDToStencilCallMap_;
}

int StencilMetaInformation::getStencilIDFromStmt(
    const std::shared_ptr<StencilCallDeclStmt>& stmt) const {
  auto it = StencilCallToStencilIDMap_.find(stmt);
  DAWN_ASSERT_MSG(it != StencilCallToStencilIDMap_.end(), "Invalid stencil call");
  return it->second;
}

std::unordered_map<std::string, int>& StencilMetaInformation::getNameToAccessIDMap() {
  return NameToAccessIDMap_;
}

const std::unordered_map<std::string, int>& StencilMetaInformation::getNameToAccessIDMap() const {
  return NameToAccessIDMap_;
}

std::unordered_map<int, std::string>& StencilMetaInformation::getAccessIDToNameMap() {
  return AccessIDToNameMap_;
}

const std::unordered_map<int, std::string>& StencilMetaInformation::getAccessIDToNameMap() const {
  return AccessIDToNameMap_;
}

std::unordered_map<int, std::string>& StencilMetaInformation::getLiteralAccessIDToNameMap() {
  return LiteralAccessIDToNameMap_;
}
const std::unordered_map<int, std::string>&
StencilMetaInformation::getLiteralAccessIDToNameMap() const {
  return LiteralAccessIDToNameMap_;
}

std::unordered_map<int, std::string>& StencilMetaInformation::getStageIDToNameMap() {
  return StageIDToNameMap_;
}

const std::unordered_map<int, std::string>& StencilMetaInformation::getStageIDToNameMap() const {
  return StageIDToNameMap_;
}

std::set<int>& StencilMetaInformation::getFieldAccessIDSet() { return FieldAccessIDSet_; }

const std::set<int>& StencilMetaInformation::getFieldAccessIDSet() const {
  return FieldAccessIDSet_;
}

std::set<int>& StencilMetaInformation::getGlobalVariableAccessIDSet() {
  return GlobalVariableAccessIDSet_;
}

const std::set<int>& StencilMetaInformation::getGlobalVariableAccessIDSet() const {
  return GlobalVariableAccessIDSet_;
}

namespace {

/// @brief Get the orignal name of the field (or variable) given by AccessID and a list of
/// SourceLocations where this field (or variable) was accessed.
class OriginalNameGetter : public ASTVisitorForwarding {
  const StencilMetaInformation* instantiation_;
  const int AccessID_;
  const bool captureLocation_;

  std::string name_;
  std::vector<SourceLocation> locations_;

public:
  OriginalNameGetter(const StencilMetaInformation* instantiation, int AccessID,
                     bool captureLocation)
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
StencilMetaInformation::getOriginalNameAndLocationsFromAccessID(
    int AccessID, const std::shared_ptr<Stmt>& stmt) const {
  OriginalNameGetter orignalNameGetter(this, AccessID, true);
  stmt->accept(orignalNameGetter);
  return orignalNameGetter.getNameLocationPair();
}

 std::string StencilMetaInformation::getOriginalNameFromAccessID(int AccessID, const IIR* iir)
 const {
  OriginalNameGetter orignalNameGetter(this, AccessID, true);

  for(const auto& stmtAccessesPair : iterateIIROver<StatementAccessesPair>(iir)) {
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

std::string StencilMetaInformation::makeLocalVariablename(const std::string& name, int AccessID) {
  return makeNameImpl("__local_", name, AccessID);
}

std::string StencilMetaInformation::makeTemporaryFieldname(const std::string& name, int AccessID) {
  return makeNameImpl("__tmp_", name, AccessID);
}

std::string StencilMetaInformation::extractLocalVariablename(const std::string& name) {
  return extractNameImpl("__local_", name);
}

std::string StencilMetaInformation::extractTemporaryFieldname(const std::string& name) {
  return extractNameImpl("__tmp_", name);
}

std::string StencilMetaInformation::makeStencilCallCodeGenName(int StencilID) {
  return "__code_gen_" + std::to_string(StencilID);
}

bool StencilMetaInformation::isStencilCallCodeGenName(const std::string& name) {
  return StringRef(name).startswith("__code_gen_");
}

const std::set<int>& StencilMetaInformation::getCachedVariableSet() const {
  return CachedVariableSet_;
}

void StencilMetaInformation::insertCachedVariable(int fieldID) {
  CachedVariableSet_.emplace(fieldID);
}

const std::unordered_map<std::string, std::shared_ptr<sir::Value>>&
StencilMetaInformation::getGlobalVariableMap() const {
  return globalVariableMap_;
}

SourceLocation& StencilMetaInformation::getStencilLocation() { return stencilLocation_; }

std::unordered_map<std::shared_ptr<Expr>, int>& StencilMetaInformation::getExprToAccessIDMap() {
  return ExprToAccessIDMap_;
}

std::set<int>& StencilMetaInformation::getTemporaryFieldAccessIDSet() {
  return TemporaryFieldAccessIDSet_;
}

std::set<int>& StencilMetaInformation::getAllocatedFieldAccessIDSet() {
    return AllocatedFieldAccessIDSet_;
}

StencilMetaInformation::VariableVersions &StencilMetaInformation::getVariableVersions()
{
    return variableVersions_;
}

std::string StencilMetaInformation::getFileName()
{
    return fileName_;
}

} // namespace iir
} // namespace dawn
