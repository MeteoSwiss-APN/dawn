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

#ifndef GSL_OPTIMIZER_PASSSTENCILSPLITTER_H
#define GSL_OPTIMIZER_PASSSTENCILSPLITTER_H

#include "gsl/Optimizer/Pass.h"

namespace gsl {

/// @brief Pass for splitting stencils due to software limitations i.e stencils are too large
///
/// This Pass depends on `PassSetStageGraph`.
///
/// @ingroup optimizer
class PassStencilSplitter : public Pass {
public:
  PassStencilSplitter();

  /// @brief Maximum number of allowed fields per stencil
  ///
  /// This is the threshold for the splitting pass to be invoked.
  static constexpr int MaxFieldPerStencil = 40;

  /// @brief Pass implementation
  bool run(StencilInstantiation* stencilInstantiation) override;
};

} // namespace gsl

#endif
