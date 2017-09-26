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

#ifndef GSL_OPTIMIZER_PASSSTAGEMERGER_H
#define GSL_OPTIMIZER_PASSSTAGEMERGER_H

#include "gsl/Optimizer/Pass.h"

namespace gsl {

/// @brief Try to merge stages into to the same Do-Method (if there are no horizontal dependencies)
/// or into seperate Do-Methods if their vertical intervals do not overlap
///
/// Merging stages is beneficial as it reduces synchronization among the threads (e.g in CUDA a
/// stage is followed by a `__syncthreads()`).
///
/// This Pass depends on `PassSetStageGraph`.
///
/// @note This pass renders the stage graphs invalid. Run `PassSetStageGraph` to compute them again.
///
/// @ingroup optimizer
class PassStageMerger : public Pass {
public:
  PassStageMerger();

  /// @brief Pass implementation
  bool run(StencilInstantiation* stencilInstantiation) override;
};

} // namespace gsl

#endif
