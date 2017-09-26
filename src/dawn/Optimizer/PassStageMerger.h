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

#ifndef DAWN_OPTIMIZER_PASSSTAGEMERGER_H
#define DAWN_OPTIMIZER_PASSSTAGEMERGER_H

#include "dawn/Optimizer/Pass.h"

namespace dawn {

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

} // namespace dawn

#endif
