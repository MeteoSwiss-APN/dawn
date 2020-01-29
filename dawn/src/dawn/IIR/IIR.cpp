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

#include "dawn/IIR/IIR.h"
#include "dawn/IIR/DependencyGraphStage.h"
#include "dawn/IIR/Field.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/IteratorAdapters.h"
#include "dawn/Support/StringUtil.h"
#include "dawn/Support/Unreachable.h"
#include <algorithm>
#include <iostream>
#include <numeric>

namespace dawn {
namespace iir {
namespace {
void mergeFields(std::unordered_map<int, Stencil::FieldInfo> const& sourceFields,
                 std::unordered_map<int, Stencil::FieldInfo>& destinationFields) {

  for(const auto& fieldPair : sourceFields) {
    Stencil::FieldInfo sField = fieldPair.second;

    auto it = destinationFields.find(fieldPair.first);
    if(it != destinationFields.end()) {
      Stencil::FieldInfo dField = destinationFields.at(fieldPair.first);

      DAWN_ASSERT(dField.Name == sField.Name);
      DAWN_ASSERT(dField.field.getFieldDimensions() == sField.field.getFieldDimensions());
      DAWN_ASSERT(dField.IsTemporary == sField.IsTemporary);

      mergeField(sField.field, dField.field);
    } else {
      destinationFields.emplace(fieldPair.first, sField);
    }
  }
}
} // namespace

const Stencil& IIR::getStencil(const int stencilID) const {
  auto lamb = [&](const std::unique_ptr<Stencil>& stencil) -> bool {
    return (stencil->getStencilID() == stencilID);
  };

  auto it = std::find_if(getChildren().begin(), getChildren().end(), lamb);
  DAWN_ASSERT(it != getChildren().end());
  return *(*it);
}

void IIR::updateFromChildren() {
  derivedInfo_.fields_.clear();

  for(const auto& stencil : children_) {
    mergeFields(stencil->getFields(), derivedInfo_.fields_);
  }
}

void IIR::DerivedInfo::clear() { fields_.clear(); }

json::json IIR::jsonDump() const {
  json::json node;

  json::json fieldsJson;
  for(const auto& f : derivedInfo_.fields_) {
    fieldsJson[f.second.Name] = f.second.jsonDump();
  }
  node["Fields"] = fieldsJson;

  int cnt = 0;
  for(const auto& stencil : children_) {
    node["Stencil" + std::to_string(cnt)] = stencil->jsonDump();
    cnt++;
  }

  json::json globalsJson;
  for(const auto& globalPair : *globalVariableMap_) {
    globalsJson[globalPair.first] = globalPair.second.jsonDump();
  }
  node["globals"] = globalsJson;

  return node;
}

IIR::IIR(const ast::GridType gridType, std::shared_ptr<sir::GlobalVariableMap> sirGlobals,
         const std::vector<std::shared_ptr<sir::StencilFunction>>& stencilFunction)
    : gridType_(gridType), globalVariableMap_(sirGlobals), stencilFunctions_(stencilFunction) {}

void IIR::clone(std::unique_ptr<IIR>& dest) const {
  dest->cloneChildrenFrom(*this, dest);
  dest->setBlockSize(blockSize_);
  dest->controlFlowDesc_ = controlFlowDesc_.clone();
  dest->globalVariableMap_ = globalVariableMap_;
}

std::unique_ptr<IIR> IIR::clone() const {
  auto cloneIIR = std::make_unique<IIR>(gridType_, globalVariableMap_, stencilFunctions_);
  clone(cloneIIR);
  return cloneIIR;
}

bool IIR::operator==(const IIR& other) const noexcept {
  // AST GridType
  if(this->gridType_ != other.gridType_)
    return false;

  // BlockSize
  for(const auto& [bs1, bs2] : zip(this->blockSize_, other.blockSize_)) {
    if(bs1 != bs2)
      return false;
  }

  // Skipping ControlFlowDescriptor

  // GlobalVariableMap
  if(*this->globalVariableMap_ != *other.globalVariableMap_)
    return false;

  // StencilFunctions
  if(!std::equal(this->stencilFunctions_.begin(), this->stencilFunctions_.end(),
                 other.stencilFunctions_.begin()))
    return false;

  // DerivedInfo
  if(!compareMapValuesAsSet(this->derivedInfo_.StageIDToNameMap_,
                            other.derivedInfo_.StageIDToNameMap_))
    return false;

  if(!compareMapValuesAsSet(this->derivedInfo_.fields_, other.derivedInfo_.fields_))
    return false;

  // Traverse downward
  if(this->getChildren().size() != other.getChildren().size())
    return false;

  for(const auto& [this_s, other_s] : zip(this->getChildren(), other.getChildren())) {
    if(*this_s != *other_s)
      return false;
  }

  return true;
}

} // namespace iir
} // namespace dawn
