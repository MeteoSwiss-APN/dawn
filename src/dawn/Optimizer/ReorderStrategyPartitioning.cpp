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
#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/IIR/StencilInstantiation.h"

namespace dawn {

std::unique_ptr<iir::Stencil>
ReoderStrategyPartitioning::reorder(const std::unique_ptr<iir::Stencil>& stencilPtr) {
  DAWN_ASSERT("ReoderStrategyPartitioning is not yet implemented");
  return nullptr;
}

} // namespace dawn
