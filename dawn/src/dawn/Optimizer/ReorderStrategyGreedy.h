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

#pragma once

#include "dawn/Optimizer/ReorderStrategy.h"

namespace dawn {

namespace iir {
class StencilInstantiation;
}
/// @brief Reordering strategy which tries to move each stage upwards as far as possible under the
/// sole constraint that the extent of any field does not exeed the maximum halo points
/// @ingroup optimizer
class ReorderStrategyGreedy : public ReorderStrategy {
public:
  /// @brief Apply the reordering strategy and return stencil
  virtual std::unique_ptr<iir::Stencil> reorder(iir::StencilInstantiation* instantiation,
                                                const std::unique_ptr<iir::Stencil>& stencil,
                                                OptimizerContext& context) override;
};

} // namespace dawn
