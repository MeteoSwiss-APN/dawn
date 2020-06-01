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

/// @brief Reordering strategy which uses S-cut graph partitioning to reorder the stages and
/// statements
/// @ingroup optimizer
class ReorderStrategyPartitioning : public ReorderStrategy {
public:
  /// @brief Apply the reordering strategy and return the stencil
  virtual std::unique_ptr<iir::Stencil> reorder(iir::StencilInstantiation* instantiation,
                                                const std::unique_ptr<iir::Stencil>& stencilPtr,
                                                OptimizerContext& context) override;
};

} // namespace dawn
