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

/// @brief Pass for splitting stencils due to software limitations i.e stencils are too large
///
/// This Pass depends on `PassSetStageGraph`.
///
/// @ingroup optimizer
///
/// This pass is not necessary to create legal code and is hence not in the debug-group
class PassStencilSplitter : public Pass {
public:
  PassStencilSplitter(OptimizerContext& context, int maxNumberOfFilelds);

  /// @brief Maximum number of allowed fields per stencil
  ///
  /// This is the threshold for the splitting pass to be invoked.
  int MaxFieldPerStencil;

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) override;
};

} // namespace dawn
