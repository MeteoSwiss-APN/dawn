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

#include "dawn/IIR/StencilMetaInformation.h"
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/SIR/AST.h"
#include "dawn/SIR/ASTStringifier.h"
#include "dawn/SIR/ASTUtil.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/FileUtil.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/Json.h"
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <stack>

namespace dawn {
namespace iir {

void StencilMetaInformation::clone(const StencilMetaInformation& origin) {
  AccessIDToNameMap_ = origin.AccessIDToNameMap_;
  for(auto pair : origin.ExprIDToAccessIDMap_) {
    ExprIDToAccessIDMap_.emplace(pair.first, pair.second);
  }
  for(auto pair : origin.StmtIDToAccessIDMap_) {
    StmtIDToAccessIDMap_.emplace(pair.first, pair.second);
  }
  LiteralAccessIDToNameMap_ = origin.LiteralAccessIDToNameMap_;
  FieldAccessIDSet_ = origin.FieldAccessIDSet_;
  apiFieldIDs_ = origin.apiFieldIDs_;
  TemporaryFieldAccessIDSet_ = origin.TemporaryFieldAccessIDSet_;
  GlobalVariableAccessIDSet_ = origin.GlobalVariableAccessIDSet_;
  for(auto id : origin.variableVersions_.getVersionIDs()) {
    variableVersions_.insert(id, origin.variableVersions_.getVersions(id));
  }
  for(const auto& sf : origin.stencilFunctionInstantiations_) {
    stencilFunctionInstantiations_.emplace_back(
        std::make_shared<StencilFunctionInstantiation>(sf->clone()));
  }
  for(const auto& pair : origin.ExprToStencilFunctionInstantiationMap_) {
    ExprToStencilFunctionInstantiationMap_.emplace(
        std::make_shared<StencilFunCallExpr>(*(pair.first)),
        std::make_shared<StencilFunctionInstantiation>(pair.second->clone()));
  }
  for(const auto& pair : origin.stencilFunInstantiationCandidate_) {
    StencilFunctionInstantiationCandidate candidate;
    candidate.callerStencilFunction_ =
        std::make_shared<StencilFunctionInstantiation>(pair.second.callerStencilFunction_->clone());
    stencilFunInstantiationCandidate_.emplace(
        std::make_shared<StencilFunctionInstantiation>(pair.first->clone()), candidate);
  }
  for(const auto& pair : origin.FieldnameToBoundaryConditionMap_) {
    FieldnameToBoundaryConditionMap_.emplace(
        pair.first, std::make_shared<BoundaryConditionDeclStmt>(*(pair.second)));
    fieldIDToInitializedDimensionsMap_ = origin.fieldIDToInitializedDimensionsMap_;
  }
  for(const auto& pair : origin.globalVariableMap_) {
    globalVariableMap_.emplace(pair.first, std::make_shared<sir::Value>(pair.second));
  }
  stencilLocation_ = origin.stencilLocation_;
  stencilName_ = origin.stencilName_;
  fileName_ = origin.fileName_;
}

json::json StencilMetaInformation::VariableVersions::jsonDump() const {
  std::unordered_map<int, std::shared_ptr<std::vector<int>>> variableVersionsMap_;
  std::unordered_map<int, int> versionToOriginalVersionMap_;
  std::unordered_set<int> versionIDs_;
  json::json node;

  json::json versionMap;
  for(const auto& pair : variableVersionsMap_) {
    json::json versions;
    for(const int id : *(pair.second)) {
      versions.push_back(id);
    }
    versionMap[std::to_string(pair.first)] = versions;
  }
  node["versions"] = versionMap;
  json::json versionID;
  for(const int id : versionIDs_) {
    versionID.push_back(id);
  }
  node["versionIDs"] = versionID;
  return node;
}

const std::string& StencilMetaInformation::getNameFromLiteralAccessID(int AccessID) const {
  DAWN_ASSERT_MSG(isLiteral(AccessID), "Invalid literal");
  return LiteralAccessIDToNameMap_.find(AccessID)->second;
}

const std::string& StencilMetaInformation::getFieldNameFromAccessID(int accessID) const {
  if(accessID < 0)
    return getNameFromLiteralAccessID(accessID);
  return AccessIDToNameMap_.directAt(accessID);
}

void StencilMetaInformation::insertAllocatedField(const int accessID) {
  AllocatedFieldAccessIDSet_.insert(accessID);
}
void StencilMetaInformation::eraseAllocatedField(const int accessID) {
  AllocatedFieldAccessIDSet_.erase(accessID);
}

bool StencilMetaInformation::isGlobalVariable(const std::string& name) const {
  auto it = getNameToAccessIDMap().find(name);
  return it == getNameToAccessIDMap().end() ? false : isGlobalVariable(it->second);
}

const std::unordered_map<std::string, int>& StencilMetaInformation::getNameToAccessIDMap() const {
  return AccessIDToNameMap_.getReverseMap();
}

/// @brief Get the AccessID-to-Name map
const std::unordered_map<int, std::string>& StencilMetaInformation::getAccessIDToNameMap() const {
  return AccessIDToNameMap_.getDirectMap();
}

// TODO what if there is no map 1 to map from name to id ?
int StencilMetaInformation::getAccessIDFromName(const std::string& name) const {
  return AccessIDToNameMap_.reverseAt(name);
}

Array3i StencilMetaInformation::getFieldDimensionsMask(int FieldID) const {
  if(fieldIDToInitializedDimensionsMap_.count(FieldID) == 0) {
    return Array3i{{1, 1, 1}};
  }
  return fieldIDToInitializedDimensionsMap_.find(FieldID)->second;
}

void StencilMetaInformation::mapExprToAccessID(const std::shared_ptr<Expr>& expr, int accessID) {
  ExprIDToAccessIDMap_.emplace(expr->getID(), accessID);
}

void StencilMetaInformation::eraseExprToAccessID(std::shared_ptr<Expr> expr) {
  DAWN_ASSERT(ExprIDToAccessIDMap_.count(expr->getID()));
  ExprIDToAccessIDMap_.erase(expr->getID());
}

void StencilMetaInformation::mapStmtToAccessID(const std::shared_ptr<Stmt>& stmt, int accessID) {
  StmtIDToAccessIDMap_.emplace(stmt->getID(), accessID);
}

std::string StencilMetaInformation::getNameFromAccessID(int accessID) const {
  if(isLiteral(accessID)) {
    return getNameFromLiteralAccessID(accessID);
  } else {
    return getFieldNameFromAccessID(accessID);
  }
}

int StencilMetaInformation::getAccessIDFromExpr(const std::shared_ptr<Expr>& expr) const {
  auto it = ExprIDToAccessIDMap_.find(expr->getID());

  DAWN_ASSERT_MSG(it != ExprIDToAccessIDMap_.end(), "Invalid Expr");
  return it->second;
}

int StencilMetaInformation::getAccessIDFromStmt(const std::shared_ptr<Stmt>& stmt) const {
  auto it = StmtIDToAccessIDMap_.find(stmt->getID());
  DAWN_ASSERT_MSG(it != StmtIDToAccessIDMap_.end(), "Invalid Stmt");
  return it->second;
}

void StencilMetaInformation::setAccessIDOfStmt(const std::shared_ptr<Stmt>& stmt,
                                               const int accessID) {
  DAWN_ASSERT(StmtIDToAccessIDMap_.count(stmt->getID()));
  StmtIDToAccessIDMap_[stmt->getID()] = accessID;
}

bool StencilMetaInformation::hasStmtToAccessID(const std::shared_ptr<Stmt>& stmt) const {
  return StmtIDToAccessIDMap_.count(stmt->getID());
}

void StencilMetaInformation::insertStmtToAccessID(const std::shared_ptr<Stmt>& stmt,
                                                  const int accessID) {
  DAWN_ASSERT(!StmtIDToAccessIDMap_.count(stmt->getID()));
  StmtIDToAccessIDMap_[stmt->getID()] = accessID;
}

void StencilMetaInformation::setAccessIDOfExpr(const std::shared_ptr<Expr>& expr,
                                               const int accessID) {
  DAWN_ASSERT(ExprIDToAccessIDMap_.count(expr->getID()));
  ExprIDToAccessIDMap_[expr->getID()] = accessID;
}

const std::shared_ptr<StencilFunctionInstantiation>
StencilMetaInformation::getStencilFunctionInstantiation(
    const std::shared_ptr<StencilFunCallExpr>& expr) const {
  auto it = ExprToStencilFunctionInstantiationMap_.find(expr);
  DAWN_ASSERT_MSG(it != ExprToStencilFunctionInstantiationMap_.end(), "Invalid stencil function");
  return it->second;
}

// TODO set or emplace ? have a convention
void StencilMetaInformation::setAccessIDNamePair(int accessID, const std::string& name) {
  AccessIDToNameMap_.emplace(accessID, name);
}

void StencilMetaInformation::setAccessIDNamePairOfField(int AccessID, const std::string& name,
                                                        bool isTemporary) {
  setAccessIDNamePair(AccessID, name);
  FieldAccessIDSet_.insert(AccessID);
  if(isTemporary) {
    TemporaryFieldAccessIDSet_.insert(AccessID);
  }
}

void StencilMetaInformation::setAccessIDNamePairOfGlobalVariable(int accessID,
                                                                 const std::string& name) {
  setAccessIDNamePair(accessID, name);
  GlobalVariableAccessIDSet_.insert(accessID);
}

void StencilMetaInformation::removeAccessID(int AccessID) {
  // TODO do we need to remove from all of them ?
  AccessIDToNameMap_.directEraseKey(AccessID);
  FieldAccessIDSet_.erase(AccessID);
  TemporaryFieldAccessIDSet_.erase(AccessID);

  if(variableVersions_.hasVariableMultipleVersions(AccessID)) {
    auto versions = variableVersions_.getVersions(AccessID);
    versions->erase(std::remove_if(versions->begin(), versions->end(),
                                   [&](int AID) { return AID == AccessID; }),
                    versions->end());
  }
}

json::json StencilMetaInformation::jsonDump() const {
  json::json metaDataJson;
  metaDataJson["VariableVersions"] = variableVersions_.jsonDump();
  size_t pos = fileName_.find_last_of("\\/");
  DAWN_ASSERT(pos + 1 < fileName_.size() - 1);
  metaDataJson["filename"] = fileName_.substr(pos + 1, fileName_.size() - pos - 1);
  metaDataJson["stencilname"] = stencilName_;
  std::stringstream ss;
  ss << stencilLocation_;
  metaDataJson["stencilLocation"] = ss.str();
  ss.str("");

  json::json globalsJson;
  for(const auto& globalPair : globalVariableMap_) {
    globalsJson[globalPair.first] = globalPair.second->jsonDump();
  }
  metaDataJson["globals"] = globalsJson;

  json::json globalAccessIDsJson;
  for(const auto& id : GlobalVariableAccessIDSet_) {
    globalAccessIDsJson.push_back(id);
  }
  metaDataJson["GlobalAccessIDs"] = globalAccessIDsJson;

  json::json fieldsMapJson;
  for(const auto& pair : fieldIDToInitializedDimensionsMap_) {
    auto dims = pair.second;
    fieldsMapJson[std::to_string(pair.first)] = format("[%i,%i,%i]", dims[0], dims[1], dims[2]);
  }
  metaDataJson["FieldDims"] = fieldsMapJson;

  json::json bcJson;
  for(const auto& bc : FieldnameToBoundaryConditionMap_) {
    bcJson[bc.first] = ASTStringifer::toString(bc.second);
  }
  metaDataJson["FieldToBC"] = bcJson;

  json::json tmpAccessIDsJson;
  for(const auto& id : TemporaryFieldAccessIDSet_) {
    tmpAccessIDsJson.push_back(id);
  }
  metaDataJson["TemporaryAccessIDs"] = tmpAccessIDsJson;

  json::json apiAccessIDsJson;
  for(const auto& id : apiFieldIDs_) {
    apiAccessIDsJson.push_back(id);
  }
  metaDataJson["apiAccessIDs"] = apiAccessIDsJson;

  json::json fieldAccessIDsJson;
  for(const auto& id : FieldAccessIDSet_) {
    fieldAccessIDsJson.push_back(id);
  }
  metaDataJson["fieldAccessIDs"] = fieldAccessIDsJson;

  json::json literalAccessIDsJson;
  for(const auto& pair : LiteralAccessIDToNameMap_) {
    literalAccessIDsJson[std::to_string(pair.first)] = pair.second;
  }
  metaDataJson["literalAccessIDs"] = literalAccessIDsJson;

  json::json accessIDToNameJson;
  for(const auto& pair : AccessIDToNameMap_.getDirectMap()) {
    accessIDToNameJson[std::to_string(pair.first)] = pair.second;
  }
  metaDataJson["AccessIDToName"] = accessIDToNameJson;
  // TODO recover?
  //  json::json idToStencilCallJson;
  //  for(const auto& pair : IDToStencilCallMap_) {
  //    idToStencilCallJson[std::to_string(pair.first)] = ASTStringifer::toString(pair.second);
  //  }
  //  metaDataJson["IDToStencilCall"] = idToStencilCallJson;
  return metaDataJson;
}

/// @brief Get the field-AccessID set
const std::set<int>& StencilMetaInformation::getFieldAccessIDSet() const {
  return FieldAccessIDSet_;
}

const std::unordered_map<std::shared_ptr<StencilCallDeclStmt>, int>&
StencilMetaInformation::getStencilCallToStencilIDMap() const {
  return StencilIDToStencilCallMap_.getReverseMap();
}
const std::unordered_map<int, std::shared_ptr<StencilCallDeclStmt>>&
StencilMetaInformation::getStencilIDToStencilCallMap() const {
  return StencilIDToStencilCallMap_.getDirectMap();
}

void StencilMetaInformation::eraseStencilCallStmt(std::shared_ptr<StencilCallDeclStmt> stmt) {
  StencilIDToStencilCallMap_.reverseEraseKey(stmt);
}
void StencilMetaInformation::eraseStencilID(const int stencilID) {
  StencilIDToStencilCallMap_.directEraseKey(stencilID);
}

int StencilMetaInformation::getStencilIDFromStencilCallStmt(
    const std::shared_ptr<StencilCallDeclStmt>& stmt) const {
  return StencilIDToStencilCallMap_.reverseAt(stmt);
}

void StencilMetaInformation::insertStencilCallStmt(std::shared_ptr<StencilCallDeclStmt> stmt,
                                                   int stencilID) {
  StencilIDToStencilCallMap_.emplace(stencilID, stmt);
}

const std::set<int>& StencilMetaInformation::getGlobalVariableAccessIDSet() const {
  return GlobalVariableAccessIDSet_;
}
const std::unordered_map<int, std::string>&
StencilMetaInformation::getLiteralAccessIDToNameMap() const {
  return LiteralAccessIDToNameMap_;
}

void StencilMetaInformation::insertLiteralAccessID(const int accessID, const std::string& name) {
  LiteralAccessIDToNameMap_.emplace(accessID, name);
}

} // namespace iir
} // namespace dawn
