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

#include "dawn/IIR/FieldAccessMetadata.h"

namespace dawn {
namespace iir {

std::string toString(FieldAccessType type) {
  switch(type) {
  case FieldAccessType::GlobalVariable:
    return "GlobalVariable";
  case FieldAccessType::Literal:
    return "Literal";
  case FieldAccessType::LocalVariable:
    return "LocalVariable";
  case FieldAccessType::StencilTemporary:
    return "StencilTemporary";
  case FieldAccessType::InterStencilTemporary:
    return "InterStencilTemporary";
  case FieldAccessType::Field:
    return "Field";
  case FieldAccessType::APIField:
    return "APIField";
  }
  return "Unkown";
}

json::json VariableVersions::jsonDump() const {
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
  for(const int id : getVersionIDs()) {
    versionID.push_back(id);
  }
  node["versionIDs"] = versionID;
  return node;
}

void FieldAccessMetadata::clone(const FieldAccessMetadata& origin) {
  LiteralAccessIDToNameMap_ = origin.LiteralAccessIDToNameMap_;
  FieldAccessIDSet_ = origin.FieldAccessIDSet_;
  apiFieldIDs_ = origin.apiFieldIDs_;
  TemporaryFieldAccessIDSet_ = origin.TemporaryFieldAccessIDSet_;
  AllocatedFieldAccessIDSet_ = origin.AllocatedFieldAccessIDSet_;
  GlobalVariableAccessIDSet_ = origin.GlobalVariableAccessIDSet_;
  for(auto idToVersionsPair : origin.variableVersions_.getvariableVersionsMap()) {
    int originalID = idToVersionsPair.first;
    for(auto versionID : *idToVersionsPair.second) {
      variableVersions_.addIDPair(originalID, versionID);
    }
  }
  accessIDType_ = origin.accessIDType_;
}
} // namespace iir
} // namespace dawn
