//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gsl/Optimizer/ReorderStrategyPartitioning.h"
#include "gsl/Optimizer/DependencyGraphAccesses.h"
#include "gsl/Optimizer/OptimizerContext.h"
#include "gsl/Optimizer/StencilInstantiation.h"

namespace gsl {

std::shared_ptr<Stencil>
ReoderStrategyPartitioning::reorder(const std::shared_ptr<Stencil>& stencilPtr) {
  GSL_ASSERT("ReoderStrategyPartitioning is not yet implemented");
  return nullptr;
}

} // namespace gsl
