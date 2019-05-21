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
#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/Optimizer/Renaming.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/StringUtil.h"
#include "dawn/Support/Unreachable.h"
#include <algorithm>
#include <iostream>
#include <numeric>

namespace dawn {
namespace iir {

std::unique_ptr<IIR> IIR::clone() const {
  auto cloneIIR = make_unique<IIR>(globalVariableMap_, stencilFunctions_);
  clone(cloneIIR);
  return cloneIIR;
}

json::json IIR::jsonDump() const {
  json::json node;

  int cnt = 0;
  for(const auto& stencil : children_) {
    node["Stencil" + std::to_string(cnt)] = stencil->jsonDump();
    cnt++;
  }

  json::json globalsJson;
  for(const auto& globalPair : globalVariableMap_) {
    globalsJson[globalPair.first] = globalPair.second->jsonDump();
  }
  node["globals"] = globalsJson;

  return node;
}

IIR::IIR(const sir::GlobalVariableMap& sirGlobals,
         const std::vector<std::shared_ptr<sir::StencilFunction>>& stencilFunction)
    : globalVariableMap_(sirGlobals), stencilFunctions_(stencilFunction) {}

void IIR::clone(std::unique_ptr<IIR>& dest) const {
  dest->cloneChildrenFrom(*this, dest);
  dest->setBlockSize(blockSize_);
  dest->controlFlowDesc_ = controlFlowDesc_.clone();
  dest->globalVariableMap_ = globalVariableMap_;
}

} // namespace iir
} // namespace dawn
