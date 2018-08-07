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

#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/DependencyGraphStage.h"
#include "dawn/Optimizer/Renaming.h"
#include "dawn/IIR/StencilInstantiation.h"
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

  cloneIIR->cloneChildren(*this, cloneIIR);
  return std::move(cloneIIR);
}

} // namespace iir
} // namespace dawn
