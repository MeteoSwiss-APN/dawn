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

#include "PassSplitStageByLocationType.h"

namespace dawn {
namespace {} // namespace

bool PassSplitStageByLocationType::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  return true;
}

} // namespace dawn
