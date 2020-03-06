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

#include "Pass.h"

namespace dawn {

/// @brief
/// * Input:
/// * Output:
/// @ingroup optimizer
///
/// This pass is necessary to generate legal IIR
///
/// Strategy:
/// - each statement goes into a single stage with location type inferred
/// Reasoning:
/// - this results in the non-optimized baseline
/// - statement/stage reordering is simple (only on stages; not statements between stages)
/// - merging after reordering is straight-forward
// TODO update description
class PassStageSplitAllStatements : public Pass {
public:
  PassStageSplitAllStatements(OptimizerContext& context)
      : Pass(context, "PassStageSplitAllStatements") {}

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) override;
};

} // namespace dawn
