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

#ifndef DAWN_OPTIMIZER_PASSSTENCILSPLITTER_H
#define DAWN_OPTIMIZER_PASSSTENCILSPLITTER_H

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
  PassStencilSplitter(int maxNumberOfFilelds);

  /// @brief Maximum number of allowed fields per stencil
  ///
  /// This is the threshold for the splitting pass to be invoked.
  int MaxFieldPerStencil;

  /// @brief Pass implementation
  bool run(const std::shared_ptr<StencilInstantiation>& stencilInstantiation) override;
};

} // namespace dawn

#endif
