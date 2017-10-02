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

#include "dawn/Optimizer/ReorderStrategyPartitioning.h"
#include "dawn/Optimizer/DependencyGraphAccesses.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/StencilInstantiation.h"

namespace dawn {

std::shared_ptr<Stencil>
ReoderStrategyPartitioning::reorder(const std::shared_ptr<Stencil>& stencilPtr) {
  DAWN_ASSERT("ReoderStrategyPartitioning is not yet implemented");
  return nullptr;
}

} // namespace dawn
