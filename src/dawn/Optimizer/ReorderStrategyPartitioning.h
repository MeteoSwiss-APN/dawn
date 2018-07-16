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

#ifndef DAWN_OPTIMIZER_REORDERSTRATEGYPARTITIONING_H
#define DAWN_OPTIMIZER_REORDERSTRATEGYPARTITIONING_H

#include "dawn/Optimizer/ReorderStrategy.h"

namespace dawn {

/// @brief Reordering strategy which uses S-cut graph partitioning to reorder the stages and
/// statements
/// @ingroup optimizer
class ReoderStrategyPartitioning : public ReorderStrategy {
public:
  /// @brief Apply the reordering strategy and return the stencil
  virtual std::shared_ptr<iir::Stencil> reorder(const std::shared_ptr<iir::Stencil>& stencilPtr) override;
};

} // namespace dawn

#endif
