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

#include "dawn/SIR/ASTStringifier.h"

namespace dawn {
namespace iir {

void StencilMetaInformation::clone(const StencilMetaInformation& origin) {
  AccessIDToNameMap_ = origin.AccessIDToNameMap_;
  for(auto pair : origin.ExprToAccessIDMap_) {
    ExprToAccessIDMap_.emplace(pair.first->clone(), pair.second);
  }
  for(auto pair : origin.StmtToAccessIDMap_) {
    StmtToAccessIDMap_.emplace(pair.first->clone(), pair.second);
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
//  for(const auto& pair : origin.BoundaryConditionToExtentsMap_) {
//    BoundaryConditionToExtentsMap_.emplace(
//        std::make_shared<BoundaryConditionDeclStmt>(*(pair.first)), pair.second);
//  }
  for(const auto& pair : origin.FieldnameToBoundaryConditionMap_) {
    FieldnameToBoundaryConditionMap_.emplace(
        pair.first, std::make_shared<BoundaryConditionDeclStmt>(*(pair.second)));
  }
//  CachedVariableSet_ = origin.CachedVariableSet_;
  fieldIDToInitializedDimensionsMap_ = origin.fieldIDToInitializedDimensionsMap_;
  for(const auto& pair : origin.globalVariableMap_) {
    globalVariableMap_.emplace(pair.first, std::make_shared<sir::Value>(pair.second));
  }
  stencilLocation_ = origin.stencilLocation_;
  stencilName_ = origin.stencilName_;
  fileName_ = origin.fileName_;
}

} // namespace iir
} // namespace dawn
