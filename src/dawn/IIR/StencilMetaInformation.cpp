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
  fieldAccessMetadata_.clone(origin.fieldAccessMetadata_);
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
  for(const auto& pair : origin.fieldnameToBoundaryConditionMap_) {
    fieldnameToBoundaryConditionMap_.emplace(
        pair.first, std::make_shared<BoundaryConditionDeclStmt>(*(pair.second)));
    fieldIDToInitializedDimensionsMap_ = origin.fieldIDToInitializedDimensionsMap_;
  }
  stencilLocation_ = origin.stencilLocation_;
  stencilName_ = origin.stencilName_;
  fileName_ = origin.fileName_;
}

const std::string& StencilMetaInformation::getNameFromLiteralAccessID(int AccessID) const {
  DAWN_ASSERT_MSG(isAccessType(iir::FieldAccessType::FAT_Literal, AccessID), "Invalid literal");
  return fieldAccessMetadata_.LiteralAccessIDToNameMap_.find(AccessID)->second;
}

const std::string& StencilMetaInformation::getFieldNameFromAccessID(int accessID) const {
  if(accessID < 0)
    return getNameFromLiteralAccessID(accessID);
  return AccessIDToNameMap_.directAt(accessID);
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

bool StencilMetaInformation::isAccessType(FieldAccessType fType, const std::string& name) const {
  if(fType == FieldAccessType::FAT_Literal) {
    throw std::runtime_error("Literal can not be queried by name");
  }
  if(!hasNameToAccessID(name))
    return false;

  return isAccessType(fType, getAccessIDFromName(name));
}
bool StencilMetaInformation::isAccessType(FieldAccessType fType, const int accessID) const {
  if(fType == FieldAccessType::FAT_Field) {
    return isAccessType(FieldAccessType::FAT_APIField, accessID) ||
           isAccessType(FieldAccessType::FAT_StencilTemporary, accessID) ||
           isAccessType(FieldAccessType::FAT_InterStencilTemporary, accessID);
  }
  if(fType == FieldAccessType::FAT_LocalVariable) {
    return !isAccessType(FieldAccessType::FAT_Field, accessID) &&
           !isAccessType(FieldAccessType::FAT_Literal, accessID) &&
           !isAccessType(FieldAccessType::FAT_GlobalVariable, accessID);
  }
  // not all the accessIDs are registered
  return (fieldAccessMetadata_.accessIDType_.count(accessID) &&
          fieldAccessMetadata_.accessIDType_.at(accessID) == fType);
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
  if(isAccessType(iir::FieldAccessType::FAT_Literal, accessID)) {
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

// TODO private ?
void StencilMetaInformation::setAccessIDNamePair(int accessID, const std::string& name) {
  AccessIDToNameMap_.emplace(accessID, name);
}

void StencilMetaInformation::insertField(FieldAccessType type, const std::string& name,
                                         const Array3i fieldDimensions) {
  int accessID = UIDGenerator::getInstance()->get();
  DAWN_ASSERT(isFieldType(type));
  insertAccessOfType(type, accessID, name);
  fieldIDToInitializedDimensionsMap_.emplace(accessID, fieldDimensions);
}

void StencilMetaInformation::removeAccessID(int AccessID) {
  AccessIDToNameMap_.directEraseKey(AccessID);

  // we can only remove fields that are not API fields (since it would make sense to transform the
  // API)
  DAWN_ASSERT(isAccessType(FieldAccessType::FAT_Field, AccessID));

  fieldAccessMetadata_.FieldAccessIDSet_.erase(AccessID);
  if(isAccessType(FieldAccessType::FAT_InterStencilTemporary, AccessID)) {
    fieldAccessMetadata_.AllocatedFieldAccessIDSet_.erase(AccessID);
  }
  if(isAccessType(FieldAccessType::FAT_StencilTemporary, AccessID)) {
    fieldAccessMetadata_.TemporaryFieldAccessIDSet_.erase(AccessID);
  }
  if(isAccessType(FieldAccessType::FAT_APIField, AccessID)) {
    // remote on a vector
    auto begin = fieldAccessMetadata_.apiFieldIDs_.begin();
    auto end = fieldAccessMetadata_.apiFieldIDs_.end();
    auto first = std::find(begin, end, AccessID);
    if(first != end) {
      for(auto i = first; ++i != end;) {
        if(!(*i == AccessID)) {
          *first++ = std::move(*i);
        }
      }
    }
  }
  fieldAccessMetadata_.accessIDType_.erase(AccessID);

  if(fieldAccessMetadata_.variableVersions_.hasVariableMultipleVersions(AccessID)) {
    auto versions = fieldAccessMetadata_.variableVersions_.getVersions(AccessID);
    versions->erase(std::remove_if(versions->begin(), versions->end(),
                                   [&](int AID) { return AID == AccessID; }),
                    versions->end());
  }
}

StencilMetaInformation::StencilMetaInformation(const sir::GlobalVariableMap& globalVariables) {
  for(const auto& global : globalVariables) {
    insertAccessOfType(iir::FieldAccessType::FAT_GlobalVariable, global.first);
  }
}

void StencilMetaInformation::insertVersions(const int accessID,
                                            std::shared_ptr<std::vector<int>> versionsID) {
  fieldAccessMetadata_.variableVersions_.insert(accessID, versionsID);
}

std::shared_ptr<std::vector<int>> StencilMetaInformation::getVersionsOf(const int accessID) const {
  return fieldAccessMetadata_.variableVersions_.getVersions(accessID);
}

json::json StencilMetaInformation::jsonDump() const {
  json::json metaDataJson;
  metaDataJson["VariableVersions"] = fieldAccessMetadata_.variableVersions_.jsonDump();
  size_t pos = fileName_.find_last_of("\\/");
  DAWN_ASSERT(pos + 1 < fileName_.size() - 1);
  metaDataJson["filename"] = fileName_.substr(pos + 1, fileName_.size() - pos - 1);
  metaDataJson["stencilname"] = stencilName_;
  std::stringstream ss;
  ss << stencilLocation_;
  metaDataJson["stencilLocation"] = ss.str();
  ss.str("");

  json::json globalAccessIDsJson;
  for(const auto& id : fieldAccessMetadata_.GlobalVariableAccessIDSet_) {
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
  for(const auto& bc : fieldnameToBoundaryConditionMap_) {
    bcJson[bc.first] = ASTStringifer::toString(bc.second);
  }
  metaDataJson["FieldToBC"] = bcJson;

  json::json accessIdToTypeJson;
  for(const auto& p : fieldAccessMetadata_.accessIDType_) {
    accessIdToTypeJson[p.first] = toString(p.second);
  }
  json::json tmpAccessIDsJson;
  for(const auto& id : fieldAccessMetadata_.TemporaryFieldAccessIDSet_) {
    tmpAccessIDsJson.push_back(id);
  }
  metaDataJson["TemporaryAccessIDs"] = tmpAccessIDsJson;

  json::json apiAccessIDsJson;
  for(const auto& id : fieldAccessMetadata_.apiFieldIDs_) {
    apiAccessIDsJson.push_back(id);
  }
  metaDataJson["apiAccessIDs"] = apiAccessIDsJson;

  json::json fieldAccessIDsJson;
  for(const auto& id : fieldAccessMetadata_.FieldAccessIDSet_) {
    fieldAccessIDsJson.push_back(id);
  }
  metaDataJson["fieldAccessIDs"] = fieldAccessIDsJson;

  json::json literalAccessIDsJson;
  for(const auto& pair : fieldAccessMetadata_.LiteralAccessIDToNameMap_) {
    literalAccessIDsJson[std::to_string(pair.first)] = pair.second;
  }
  metaDataJson["literalAccessIDs"] = literalAccessIDsJson;

  json::json accessIDToNameJson;
  for(const auto& pair : AccessIDToNameMap_.getDirectMap()) {
    accessIDToNameJson[std::to_string(pair.first)] = pair.second;
  }
  metaDataJson["AccessIDToName"] = accessIDToNameJson;
  return metaDataJson;
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

} // namespace iir
} // namespace dawn
