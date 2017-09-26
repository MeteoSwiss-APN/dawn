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

#ifndef GSL_OPTIMIZER_PASSSTAGEREORDERING_H
#define GSL_OPTIMIZER_PASSSTAGEREORDERING_H

#include "gsl/Optimizer/Pass.h"
#include "gsl/Optimizer/ReorderStrategy.h"

namespace gsl {

/// @brief Pass for reordering stages to increase data locility
///
/// This Pass depends on `PassSetStageGraph`.
///
/// @ingroup optimizer
class PassStageReordering : public Pass {
public:
  PassStageReordering(ReorderStrategy::ReorderStrategyKind strategy);

  /// @brief Pass implementation
  bool run(StencilInstantiation* stencilInstantiation) override;

private:
  ReorderStrategy::ReorderStrategyKind strategy_;
};

} // namespace gsl

#endif
