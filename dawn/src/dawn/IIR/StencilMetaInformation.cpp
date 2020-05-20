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
#include "dawn/IIR/AST.h"
#include "dawn/IIR/ASTStringifier.h"
#include "dawn/IIR/ASTUtil.h"
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/IIR/InstantiationHelper.h"
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/Json.h"

#include <cstdlib>
#include <fstream>
#include <functional>
#include <stack>

namespace dawn {
namespace iir {

void StencilMetaInformation::clone(const StencilMetaInformation& origin) {
  AccessIDToNameMap_ = origin.AccessIDToNameMap_;
  fieldAccessMetadata_.clone(origin.fieldAccessMetadata_);
  for(const auto& sf : origin.stencilFunctionInstantiations_) {
    stencilFunctionInstantiations_.emplace_back(
        std::make_shared<StencilFunctionInstantiation>(sf->clone()));
  }
  for(const auto& pair : origin.ExprToStencilFunctionInstantiationMap_) {
    ExprToStencilFunctionInstantiationMap_.emplace(
        std::make_shared<iir::StencilFunCallExpr>(*(pair.first)),
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
        pair.first, std::make_shared<ast::BoundaryConditionDeclStmt>(*(pair.second)));
    fieldIDToInitializedDimensionsMap_ = origin.fieldIDToInitializedDimensionsMap_;
  }
  stencilLocation_ = origin.stencilLocation_;
  stencilName_ = origin.stencilName_;
  fileName_ = origin.fileName_;
}

const std::string& StencilMetaInformation::getNameFromLiteralAccessID(int AccessID) const {
  DAWN_ASSERT_MSG(isAccessType(iir::FieldAccessType::Literal, AccessID), "Invalid literal");
  return fieldAccessMetadata_.LiteralAccessIDToNameMap_.find(AccessID)->second;
}

const std::string& StencilMetaInformation::getFieldNameFromAccessID(int accessID) const {
  if(accessID < 0)
    return getNameFromLiteralAccessID(accessID);
  DAWN_ASSERT_MSG(AccessIDToNameMap_.getDirectMap().count(accessID), "Unregistered access id");
  return AccessIDToNameMap_.directAt(accessID);
}

const std::unordered_map<std::string, int>& StencilMetaInformation::getNameToAccessIDMap() const {
  return AccessIDToNameMap_.getReverseMap();
}

/// @brief Get the AccessID-to-Name map
const std::unordered_map<int, std::string>& StencilMetaInformation::getAccessIDToNameMap() const {
  return AccessIDToNameMap_.getDirectMap();
}

int StencilMetaInformation::getAccessIDFromName(const std::string& name) const {
  return AccessIDToNameMap_.reverseAt(name);
}

bool StencilMetaInformation::isAccessType(FieldAccessType fType, const std::string& name) const {
  DAWN_ASSERT_MSG(fType != FieldAccessType::Literal, "Literal can not be queried by name");
  if(!hasNameToAccessID(name))
    return false;

  return isAccessType(fType, getAccessIDFromName(name));
}
void StencilMetaInformation::markStencilFunctionInstantiationFinal(
    const std::shared_ptr<StencilFunctionInstantiation>& stencilFun) {
  stencilFunInstantiationCandidate_.erase(stencilFun);
  stencilFunctionInstantiations_.push_back(stencilFun);
}

void StencilMetaInformation::deregisterStencilFunction(
    std::shared_ptr<StencilFunctionInstantiation> stencilFun) {

  bool found = RemoveIf(ExprToStencilFunctionInstantiationMap_,
                        [&](std::pair<std::shared_ptr<iir::StencilFunCallExpr>,
                                      std::shared_ptr<StencilFunctionInstantiation>>
                                pair) { return (pair.second == stencilFun); });
  DAWN_ASSERT(found);
  found = RemoveIf(
      stencilFunctionInstantiations_,
      [&](const std::shared_ptr<StencilFunctionInstantiation>& v) { return (v == stencilFun); });
  DAWN_ASSERT(found);
}

std::shared_ptr<StencilFunctionInstantiation> StencilMetaInformation::cloneStencilFunctionCandidate(
    const std::shared_ptr<StencilFunctionInstantiation>& stencilFun, std::string functionName) {
  DAWN_ASSERT(getStencilFunInstantiationCandidates().count(stencilFun));
  auto stencilFunClone = std::make_shared<StencilFunctionInstantiation>(stencilFun->clone());

  auto stencilFunExpr =
      std::dynamic_pointer_cast<iir::StencilFunCallExpr>(stencilFun->getExpression()->clone());
  stencilFunExpr->setCallee(functionName);

  auto sirStencilFun = std::make_shared<sir::StencilFunction>(*(stencilFun->getStencilFunction()));
  sirStencilFun->Name = functionName;

  stencilFunClone->setExpression(stencilFunExpr);
  stencilFunClone->setStencilFunction(sirStencilFun);

  addStencilFunInstantiationCandidate(stencilFunClone,
                                      getStencilFunInstantiationCandidates().at(stencilFun));
  return stencilFunClone;
}

void StencilMetaInformation::removeStencilFunctionInstantiation(
    const std::shared_ptr<iir::StencilFunCallExpr>& expr,
    std::shared_ptr<StencilFunctionInstantiation> callerStencilFunctionInstantiation) {

  std::shared_ptr<StencilFunctionInstantiation> func = nullptr;

  if(callerStencilFunctionInstantiation) {
    func = callerStencilFunctionInstantiation->getStencilFunctionInstantiation(expr);
    callerStencilFunctionInstantiation->removeStencilFunctionInstantiation(expr);
  } else {
    func = getStencilFunctionInstantiation(expr);
    eraseExprToStencilFunction(expr);
  }

  eraseStencilFunctionInstantiation(func);
}

std::shared_ptr<StencilFunctionInstantiation>
StencilMetaInformation::getStencilFunctionInstantiationCandidate(
    const std::shared_ptr<iir::StencilFunCallExpr>& expr) {
  const auto& candidates = getStencilFunInstantiationCandidates();
  auto it = std::find_if(
      candidates.begin(), candidates.end(),
      [&](std::pair<std::shared_ptr<StencilFunctionInstantiation>,
                    StencilMetaInformation::StencilFunctionInstantiationCandidate> const& pair) {
        return (pair.first->getExpression() == expr);
      });
  DAWN_ASSERT_MSG((it != candidates.end()), "stencil function candidate not found");

  return it->first;
}

std::shared_ptr<StencilFunctionInstantiation>
StencilMetaInformation::getStencilFunctionInstantiationCandidate(const std::string stencilFunName,
                                                                 const Interval& interval) {
  const auto& candidates = getStencilFunInstantiationCandidates();
  auto it = std::find_if(
      candidates.begin(), candidates.end(),
      [&](std::pair<std::shared_ptr<StencilFunctionInstantiation>,
                    StencilMetaInformation::StencilFunctionInstantiationCandidate> const& pair) {
        return (pair.first->getExpression()->getCallee() == stencilFunName &&
                (pair.first->getInterval() == interval));
      });
  DAWN_ASSERT_MSG((it != candidates.end()), "stencil function candidate not found");

  return it->first;
}

void StencilMetaInformation::finalizeStencilFunctionSetup(
    std::shared_ptr<StencilFunctionInstantiation> stencilFun) {

  DAWN_ASSERT(getStencilFunInstantiationCandidates().count(stencilFun));
  stencilFun->closeFunctionBindings();
  // We take the candidate to stencil function and placed it in the stencil function instantiations
  // container
  StencilMetaInformation::StencilFunctionInstantiationCandidate candidate =
      getStencilFunInstantiationCandidates().at(stencilFun);

  // map of expr to stencil function instantiation is updated
  if(candidate.callerStencilFunction_) {
    candidate.callerStencilFunction_->insertExprToStencilFunction(stencilFun);
  } else {
    insertExprToStencilFunctionInstantiation(stencilFun);
  }

  // we move the candidate to stencil function to a final stencil function
  markStencilFunctionInstantiationFinal(stencilFun);
}

void StencilMetaInformation::insertExprToStencilFunctionInstantiation(
    const std::shared_ptr<StencilFunctionInstantiation>& stencilFun) {
  insertExprToStencilFunctionInstantiation(stencilFun->getExpression(), stencilFun);
}

bool StencilMetaInformation::isAccessType(FieldAccessType fType, const int accessID) const {
  if(fType == FieldAccessType::Field) {
    return isAccessType(FieldAccessType::APIField, accessID) ||
           isAccessType(FieldAccessType::StencilTemporary, accessID) ||
           isAccessType(FieldAccessType::InterStencilTemporary, accessID);
  }
  if(fType == FieldAccessType::LocalVariable) {
    return !isAccessType(FieldAccessType::Field, accessID) &&
           !isAccessType(FieldAccessType::Literal, accessID) &&
           !isAccessType(FieldAccessType::GlobalVariable, accessID);
  }
  if(fType == FieldAccessType::Literal) {
    return fieldAccessMetadata_.LiteralAccessIDToNameMap_.count(accessID) > 0;
  }
  // not all the accessIDs are registered
  return (fieldAccessMetadata_.accessIDType_.count(accessID) &&
          fieldAccessMetadata_.accessIDType_.at(accessID) == fType);
}

void StencilMetaInformation::moveRegisteredFieldTo(FieldAccessType type, int accessID) {
  // we can not move it into an API field, since the original order would not be preserved
  DAWN_ASSERT(type != FieldAccessType::APIField);
  DAWN_ASSERT_MSG(isFieldType(type), "non field access type can not be moved");

  fieldAccessMetadata_.accessIDType_[accessID] = type;

  if(fieldAccessMetadata_.TemporaryFieldAccessIDSet_.count(accessID)) {
    fieldAccessMetadata_.TemporaryFieldAccessIDSet_.erase(accessID);
  }
  if(fieldAccessMetadata_.AllocatedFieldAccessIDSet_.count(accessID)) {
    fieldAccessMetadata_.AllocatedFieldAccessIDSet_.erase(accessID);
  }

  if(type == FieldAccessType::StencilTemporary) {
    fieldAccessMetadata_.TemporaryFieldAccessIDSet_.insert(accessID);
  } else if(type == FieldAccessType::InterStencilTemporary) {
    fieldAccessMetadata_.AllocatedFieldAccessIDSet_.insert(accessID);
  }
}

int StencilMetaInformation::insertAccessOfType(FieldAccessType type, const std::string& name) {
  int accessID = UIDGenerator::getInstance()->get();
  if(type == FieldAccessType::Literal)
    accessID = -accessID;
  insertAccessOfType(type, accessID, name);
  return accessID;
}

void StencilMetaInformation::insertAccessOfType(FieldAccessType type, int AccessID,
                                                const std::string& name) {
  if(type != FieldAccessType::Literal) {
    addAccessIDNamePair(AccessID, name);
    fieldAccessMetadata_.accessIDType_[AccessID] = type;
  }

  if(isFieldType(type)) {
    fieldAccessMetadata_.FieldAccessIDSet_.insert(AccessID);
    if(type == FieldAccessType::StencilTemporary) {
      fieldAccessMetadata_.TemporaryFieldAccessIDSet_.insert(AccessID);
    } else if(type == FieldAccessType::InterStencilTemporary) {
      fieldAccessMetadata_.AllocatedFieldAccessIDSet_.insert(AccessID);
    } else if(type == FieldAccessType::APIField) {
      fieldAccessMetadata_.apiFieldIDs_.push_back(AccessID);
    }
  } else if(type == FieldAccessType::GlobalVariable) {
    fieldAccessMetadata_.GlobalVariableAccessIDSet_.insert(AccessID);
  } else if(type == FieldAccessType::LocalVariable) {
    // local variables are not stored
  } else if(type == FieldAccessType::Literal) {
    DAWN_ASSERT(AccessID < 0);
    fieldAccessMetadata_.LiteralAccessIDToNameMap_.emplace(AccessID, name);
  }
}

int StencilMetaInformation::addStmt(bool keepVarName, const std::shared_ptr<VarDeclStmt>& stmt) {
  int accessID = UIDGenerator::getInstance()->get();

  std::string globalName;
  if(keepVarName) {
    globalName = stmt->getName();
  } else {
    globalName = InstantiationHelper::makeLocalVariablename(stmt->getName(), accessID);
  }

  addAccessIDNamePair(accessID, globalName);
  // Add empty data object for local variable
  addAccessIDToLocalVariableDataPair(accessID, LocalVariableData{});

  DAWN_ASSERT(!stmt->getData<iir::VarDeclStmtData>().AccessID);
  stmt->getData<iir::VarDeclStmtData>().AccessID = std::make_optional(accessID);

  return accessID;
}

std::shared_ptr<VarDeclStmt>
StencilMetaInformation::declareVar(bool keepVarName, std::string varName, Type type, int accessID) {
  return declareVar(keepVarName, varName, type, nullptr, accessID);
}
std::shared_ptr<VarDeclStmt> StencilMetaInformation::declareVar(bool keepVarName,
                                                                std::string varName, Type type,
                                                                std::shared_ptr<Expr> rhs,
                                                                int accessID) {
  // TODO: find a way to reuse code from addStmt
  std::string globalName =
      keepVarName ? varName : InstantiationHelper::makeLocalVariablename(varName, accessID);
  // Construct the variable declaration
  std::vector<std::shared_ptr<iir::Expr>> initList;
  if(rhs != nullptr) { // If nullptr, declaration without initialization (overload)
    initList.push_back(rhs);
  }
  auto varDeclStmt = iir::makeVarDeclStmt(type, globalName, 0, "=", std::move(initList));
  // Update id to name map
  addAccessIDNamePair(accessID, globalName);
  // Add empty data object for local variable
  addAccessIDToLocalVariableDataPair(accessID, LocalVariableData{});
  // Update varDeclStmt's data
  varDeclStmt->getData<iir::VarDeclStmtData>().AccessID = std::make_optional(accessID);

  return varDeclStmt;
}

bool StencilMetaInformation::isFieldType(FieldAccessType accessType) const {
  return accessType == FieldAccessType::Field || accessType == FieldAccessType::APIField ||
         accessType == FieldAccessType::StencilTemporary ||
         accessType == FieldAccessType::InterStencilTemporary;
}

sir::FieldDimensions StencilMetaInformation::getFieldDimensions(int fieldID) const {
  if(isAccessIDAVersion(fieldID)) {
    fieldID = getOriginalVersionOfAccessID(fieldID);
  }
  DAWN_ASSERT_MSG(fieldIDToInitializedDimensionsMap_.count(fieldID) != 0,
                  "Field id does not exist");
  return fieldIDToInitializedDimensionsMap_.find(fieldID)->second;
}

void StencilMetaInformation::setFieldDimensions(int fieldID,
                                                sir::FieldDimensions&& fieldDimensions) {
  fieldIDToInitializedDimensionsMap_.emplace(fieldID, std::move(fieldDimensions));
}

std::string StencilMetaInformation::getNameFromAccessID(int accessID) const {
  if(isAccessType(iir::FieldAccessType::Literal, accessID)) {
    return getNameFromLiteralAccessID(accessID);
  } else {
    return getFieldNameFromAccessID(accessID);
  }
}

const std::shared_ptr<StencilFunctionInstantiation>
StencilMetaInformation::getStencilFunctionInstantiation(
    const std::shared_ptr<iir::StencilFunCallExpr>& expr) const {
  auto it = ExprToStencilFunctionInstantiationMap_.find(expr);
  DAWN_ASSERT_MSG(it != ExprToStencilFunctionInstantiationMap_.end(), "Invalid stencil function");
  return it->second;
}

void StencilMetaInformation::addAccessIDNamePair(int accessID, const std::string& name) {
  // this fails if -fkeep-varnames is used
  AccessIDToNameMap_.add(accessID, name);
}

int StencilMetaInformation::addField(FieldAccessType type, const std::string& name,
                                     sir::FieldDimensions&& fieldDimensions,
                                     std::optional<int> accessID) {
  if(!accessID.has_value()) {
    accessID = UIDGenerator::getInstance()->get();
  }
  DAWN_ASSERT(isFieldType(type));
  insertAccessOfType(type, *accessID, name);

  DAWN_ASSERT(!fieldIDToInitializedDimensionsMap_.count(*accessID));
  fieldIDToInitializedDimensionsMap_.emplace(*accessID, std::move(fieldDimensions));

  return *accessID;
}

int StencilMetaInformation::addTmpField(FieldAccessType type, const std::string& basename,
                                        sir::FieldDimensions&& fieldDimensions,
                                        std::optional<int> accessID) {
  if(!accessID.has_value()) {
    accessID = UIDGenerator::getInstance()->get();
  }

  std::string fname = InstantiationHelper::makeTemporaryFieldname(basename, *accessID);

  DAWN_ASSERT(isFieldType(type));
  insertAccessOfType(type, *accessID, fname);

  DAWN_ASSERT(!fieldIDToInitializedDimensionsMap_.count(*accessID));
  fieldIDToInitializedDimensionsMap_.emplace(*accessID, std::move(fieldDimensions));

  return *accessID;
}
ast::LocationType StencilMetaInformation::getDenseLocationTypeFromAccessID(int AccessID) const {
  DAWN_ASSERT_MSG(
      sir::dimension_isa<sir::UnstructuredFieldDimension>(
          fieldIDToInitializedDimensionsMap_.at(AccessID).getHorizontalFieldDimension()),
      "Location type requested for Cartesian dimension");
  const auto& dim = sir::dimension_cast<sir::UnstructuredFieldDimension const&>(
      fieldIDToInitializedDimensionsMap_.at(AccessID).getHorizontalFieldDimension());
  return dim.getDenseLocationType();
}

void StencilMetaInformation::removeAccessID(int AccessID) {
  AccessIDToNameMap_.directEraseKey(AccessID);

  // we can only remove field or local variables (i.e. we can not remove neither globals nor
  // literals
  DAWN_ASSERT(isAccessType(FieldAccessType::Field, AccessID) ||
              isAccessType(FieldAccessType::LocalVariable, AccessID));

  if(isAccessType(FieldAccessType::LocalVariable, AccessID)) {
    accessIDToLocalVariableDataMap_.erase(AccessID);
  }

  fieldAccessMetadata_.FieldAccessIDSet_.erase(AccessID);
  if(isAccessType(FieldAccessType::InterStencilTemporary, AccessID)) {
    fieldAccessMetadata_.AllocatedFieldAccessIDSet_.erase(AccessID);
  }
  if(isAccessType(FieldAccessType::StencilTemporary, AccessID)) {
    fieldAccessMetadata_.TemporaryFieldAccessIDSet_.erase(AccessID);
  }
  if(isAccessType(FieldAccessType::APIField, AccessID)) {
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

  if(fieldAccessMetadata_.variableVersions_.variableHasMultipleVersions(AccessID)) {
    auto versions = fieldAccessMetadata_.variableVersions_.getVersions(AccessID);
    versions->erase(std::remove_if(versions->begin(), versions->end(),
                                   [&](int AID) { return AID == AccessID; }),
                    versions->end());
  }
}

StencilMetaInformation::StencilMetaInformation(
    std::shared_ptr<sir::GlobalVariableMap> globalVariables) {
  for(const auto& global : *globalVariables) {
    insertAccessOfType(iir::FieldAccessType::GlobalVariable, global.first);
  }
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
    fieldsMapJson[std::to_string(pair.first)] = dims.toString();
  }
  metaDataJson["FieldDims"] = fieldsMapJson;

  json::json bcJson;
  for(const auto& bc : fieldnameToBoundaryConditionMap_) {
    bcJson[bc.first] = ASTStringifier::toString(bc.second);
  }
  metaDataJson["FieldToBC"] = bcJson;

  json::json accessIdToTypeJson;
  for(const auto& p : fieldAccessMetadata_.accessIDType_) {
    accessIdToTypeJson[std::to_string(p.first)] = toString(p.second);
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

  json::json idToStencilCallJson;
  for(const auto& pair : StencilIDToStencilCallMap_.getDirectMap()) {
    idToStencilCallJson[std::to_string(pair.first)] = ASTStringifier::toString(pair.second);
  }
  metaDataJson["IDToStencilCall"] = idToStencilCallJson;

  return metaDataJson;
}

const std::unordered_map<std::shared_ptr<iir::StencilCallDeclStmt>, int>&
StencilMetaInformation::getStencilCallToStencilIDMap() const {
  return StencilIDToStencilCallMap_.getReverseMap();
}
const std::unordered_map<int, std::shared_ptr<iir::StencilCallDeclStmt>>&
StencilMetaInformation::getStencilIDToStencilCallMap() const {
  return StencilIDToStencilCallMap_.getDirectMap();
}

void StencilMetaInformation::eraseStencilCallStmt(std::shared_ptr<iir::StencilCallDeclStmt> stmt) {
  StencilIDToStencilCallMap_.reverseEraseKey(stmt);
}
void StencilMetaInformation::eraseStencilID(const int stencilID) {
  StencilIDToStencilCallMap_.directEraseKey(stencilID);
}

int StencilMetaInformation::getStencilIDFromStencilCallStmt(
    const std::shared_ptr<iir::StencilCallDeclStmt>& stmt) const {
  return StencilIDToStencilCallMap_.reverseAt(stmt);
}

void StencilMetaInformation::addStencilCallStmt(std::shared_ptr<StencilCallDeclStmt> stmt,
                                                int stencilID) {
  StencilIDToStencilCallMap_.add(stencilID, stmt);
}

void StencilMetaInformation::addAccessIDToLocalVariableDataPair(int accessID,
                                                                LocalVariableData&& data) {
  DAWN_ASSERT(isAccessType(FieldAccessType::LocalVariable, accessID));
  accessIDToLocalVariableDataMap_.emplace(accessID, std::move(data));
}

iir::LocalVariableData& StencilMetaInformation::getLocalVariableDataFromAccessID(int accessID) {
  DAWN_ASSERT(isAccessType(FieldAccessType::LocalVariable, accessID));
  DAWN_ASSERT(accessIDToLocalVariableDataMap_.count(accessID));
  return accessIDToLocalVariableDataMap_.at(accessID);
}

const iir::LocalVariableData&
StencilMetaInformation::getLocalVariableDataFromAccessID(int accessID) const {
  DAWN_ASSERT(isAccessType(FieldAccessType::LocalVariable, accessID));
  DAWN_ASSERT(accessIDToLocalVariableDataMap_.count(accessID));
  return accessIDToLocalVariableDataMap_.at(accessID);
}

void StencilMetaInformation::resetLocalVarTypes() {
  for(auto& pair : accessIDToLocalVariableDataMap_) {
    pair.second = iir::LocalVariableData{};
  }
}

} // namespace iir
} // namespace dawn
