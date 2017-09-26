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

#ifndef GSL_OPTIMIZER_REORDERSTRATEGYPARTITIONING_H
#define GSL_OPTIMIZER_REORDERSTRATEGYPARTITIONING_H

#include "gsl/Optimizer/ReorderStrategy.h"

namespace gsl {

/// @brief Reordering strategy which uses S-cut graph partitioning to reorder the stages and
/// statements
/// @ingroup optimizer
class ReoderStrategyPartitioning : public ReorderStrategy {
public:
  /// @brief Apply the reordering strategy and return the stencil
  virtual std::shared_ptr<Stencil> reorder(const std::shared_ptr<Stencil>& stencilPtr) override;
};

} // namespace gsl

#endif
