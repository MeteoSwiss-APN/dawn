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
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/Optimizer/AccessComputation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassTemporaryType.h"
#include "dawn/Optimizer/Renaming.h"
#include "dawn/Optimizer/Replacing.h"
#include "dawn/Optimizer/StatementMapper.h"
#include "dawn/SIR/AST.h"
#include "dawn/SIR/ASTStringifier.h"
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
  for(auto statement : origin.stencilDescStatements_) {
    stencilDescStatements_.emplace_back(statement->clone());
  }
  for(const auto& pair : origin.IDToStencilCallMap_) {
    IDToStencilCallMap_.emplace(pair.first, std::make_shared<StencilCallDeclStmt>(*(pair.second)));
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
    for(const auto& pair : origin.globalVariableMap_) {
      globalVariableMap_.emplace(pair.first, std::make_shared<sir::Value>(pair.second));
    }
    stencilLocation_ = origin.stencilLocation_;
    stencilName_ = origin.stencilName_;
    fileName_ = origin.fileName_;
  }
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
    versionMap[pair.first] = versions;
  }
  node["versions"] = versionMap;
  json::json versionID;
  for(const int id : versionIDs_) {
    versionID.push_back(id);
  }
  node["versionIDs"] = versionID;
  return node;
}

json::json StencilMetaInformation::jsonDump() const {

  //  /// Surjection of AST Nodes, Expr (FieldAccessExpr or VarAccessExpr) or Stmt
  //  (VarDeclStmt), to
  //  /// their AccessID. The surjection implies that multiple AST Nodes can have the same
  //  AccessID,
  //  /// which is the intended behaviour as we want to get the same ID back when we access the
  //  same
  //  /// field for example
  //  std::unordered_map<int, int> ExprIDToAccessIDMap_;
  //  std::unordered_map<int, int> StmtIDToAccessIDMap_;

  //  /// Stencil description statements. These are built from the StencilDescAst of the
  //  sir::Stencil
  //  std::vector<std::shared_ptr<Statement>> stencilDescStatements_;
  //  std::unordered_map<int, std::shared_ptr<StencilCallDeclStmt>> IDToStencilCallMap_;

  //  /// Referenced stencil functions in this stencil (note that nested stencil functions are
  //  not
  //  /// stored here but rather in the respecticve `StencilFunctionInstantiation`)
  //  std::vector<std::shared_ptr<StencilFunctionInstantiation>> stencilFunctionInstantiations_;
  //  std::unordered_map<std::shared_ptr<StencilFunCallExpr>,
  //                     std::shared_ptr<StencilFunctionInstantiation>>
  //      ExprToStencilFunctionInstantiationMap_;

  //  // TODO a set here would be enough
  //  /// lookup table containing all the stencil function candidates, whose arguments are not
  //  yet
  //  bound
  //  std::unordered_map<std::shared_ptr<StencilFunctionInstantiation>,
  //                     StencilFunctionInstantiationCandidate>
  //      stencilFunInstantiationCandidate_;

  //  std::vector<std::shared_ptr<sir::StencilFunction>> allStencilFunctions_;

  json::json node;
  node["VariableVersions"] = variableVersions_.jsonDump();
  node["filename"] = fileName_;
  node["stencilname"] = stencilName_;
  std::stringstream ss;
  ss << stencilLocation_;
  node["stencilLocation"] = ss.str();
  ss.str("");

  json::json globalsJson;
  for(const auto& globalPair : globalVariableMap_) {
    globalsJson[globalPair.first] = globalPair.second->jsonDump();
  }
  node["globals"] = globalsJson;

  json::json globalAccessIDsJson;
  for(const auto& id : GlobalVariableAccessIDSet_) {
    globalAccessIDsJson.push_back(id);
  }
  node["GlobalAccessIDs"] = globalAccessIDsJson;

  json::json fieldsMapJson;
  for(const auto& pair : fieldIDToInitializedDimensionsMap_) {
    auto dims = pair.second;
    fieldsMapJson[pair.first] = format("[%i,%i,%i]", dims[0], dims[1], dims[2]);
  }
  node["FieldDims"] = fieldsMapJson;

  json::json bcJson;
  for(const auto& bc : FieldnameToBoundaryConditionMap_) {
    bcJson[bc.first] = ASTStringifer::toString(bc.second);
  }
  node["FieldToBC"] = bcJson;
  return node;

  json::json tmpAccessIDsJson;
  for(const auto& id : TemporaryFieldAccessIDSet_) {
    tmpAccessIDsJson.push_back(id);
  }
  node["TemporaryAccessIDs"] = tmpAccessIDsJson;

  json::json apiAccessIDsJson;
  for(const auto& id : apiFieldIDs_) {
    apiAccessIDsJson.push_back(id);
  }
  node["apiAccessIDs"] = apiAccessIDsJson;

  json::json fieldAccessIDsJson;
  for(const auto& id : FieldAccessIDSet_) {
    fieldAccessIDsJson.push_back(id);
  }
  node["fieldAccessIDs"] = fieldAccessIDsJson;

  json::json literalAccessIDsJson;
  for(const auto& pair : LiteralAccessIDToNameMap_) {
    fieldAccessIDsJson[pair.first] = pair.second;
  }
  node["literalAccessIDs"] = literalAccessIDsJson;

  json::json accessIDToNameJson;
  for(const auto& pair : AccessIDToNameMap_) {
    accessIDToNameJson[pair.first] = accessIDToNameJson[pair.second];
  }
  node["AccessIDToName"] = accessIDToNameJson;

  json::json idToStencilCallJson;
  for(const auto& pair : IDToStencilCallMap_) {
    idToStencilCallJson[pair.first] = ASTStringifer::toString(pair.second);
  }
  node["IDToStencilCall"] = idToStencilCallJson;
}

} // namespace iir
} // namespace dawn
