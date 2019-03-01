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

#include "dawn/IIR/DependencyGraphStage.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilInstantiation.h"
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
  auto cloneIIR = make_unique<IIR>();
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
  return node;
}

void IIR::clone(std::unique_ptr<IIR>& dest) const {
  dest->cloneChildrenFrom(*this, dest);
  dest->setBlockSize(blockSize_);
  // notice the control flow is not deep cloned, but copied
  dest->controlFlowDesc_ = controlFlowDesc_;
}

} // namespace iir
} // namespace dawn
