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

#ifndef GSL_OPTIMIZER_REORDERSTRATEGYGREEDY_H
#define GSL_OPTIMIZER_REORDERSTRATEGYGREEDY_H

#include "gsl/Optimizer/ReorderStrategy.h"

namespace gsl {

/// @brief Reordering strategy which tries to move each stage upwards as far as possible under the
/// sole constraint that the extent of any field does not exeed the maximum halo points
/// @ingroup optimizer
class ReoderStrategyGreedy : public ReorderStrategy {
public:
  /// @brief Apply the reordering strategy and return stencil
  virtual std::shared_ptr<Stencil> reorder(const std::shared_ptr<Stencil>& stencilPtr) override;
};

} // namespace gsl

#endif
