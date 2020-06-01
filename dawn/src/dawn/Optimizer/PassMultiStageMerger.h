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

#include "dawn/IIR/DependencyGraphStage.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/Optimizer/Pass.h"

namespace dawn {

/// @brief Try to merge multistages to reduce the amount of synchronization required.
///
/// This Pass depends on `PassSetStageGraph`.
///
/// @note This pass renders the stage graphs invalid. Run `PassSetStageGraph` to compute them again.
///
/// @ingroup optimizer
///
/// This pass is not necessary to create legal code and is hence not in the debug-group
class PassMultiStageMerger : public Pass {
public:
  PassMultiStageMerger(OptimizerContext& context);

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) override;
};

} // namespace dawn
