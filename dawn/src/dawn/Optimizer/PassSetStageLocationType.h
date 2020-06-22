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

#include "dawn/Optimizer/Pass.h"

namespace dawn {

/// @brief
/// * Input: any IIR where GridType == Unstructured and where statements inside the same stage are
/// on the same location type (local variables must have a location type and no scalar variables are
/// allowed).
/// * Output: same IIR, but each stage will have it's location type correctly set (!= std::nullopt)
/// to the location type of its statements
/// @ingroup optimizer
///
/// This pass is necessary to generate legal IIR for GridType == Unstructured.
///
/// Implementation detail:
/// - Uses the first statement (which has location type information) to deduce the location type.
/// - The input requirement of all statements having the same location type can be trivially ensured
///   by splitting each statement into a separate stage (see PassStageSplitAllStatements).
///   -> this results in the non-optimized baseline
///   Reasoning:
///   - statement/stage reordering is simple (only on stages; not statements between stages)
///   - merging after reordering is straight-forward
class PassSetStageLocationType : public Pass {
public:
  PassSetStageLocationType() : Pass("PassSetStageLocationType") {}

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
           const Options& options = {}) override;
};

} // namespace dawn
