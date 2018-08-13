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

#ifndef DAWN_OPTIMIZER_PASSCOMPUTESTAGEEXTENTS_H
#define DAWN_OPTIMIZER_PASSCOMPUTESTAGEEXTENTS_H

// ===-------------------------------------------------------------------------------------------===
//      The Algorithm in this pass was broken and is now replaced by a greedier one that works
//        in order to fix it, Point One and Two [ExtentComputationTODO] needs to be optimized:
//        https://github.com/MeteoSwiss-APN/dawn/issues/104
// ===-------------------------------------------------------------------------------------------===
#include "dawn/Optimizer/Pass.h"

namespace dawn {

/// @brief This Pass computes the extents (associated to redundant computations) of each stage
/// The pass takes as input a collection of stages of each multi-stage from the StencilInstantation
/// and stores the computation in the `Extent` member of the Stage (@see Stage)
///
/// This Pass needs to be recomputed if the collection and order of stages/multistages changes
///
/// @ingroup optimizer
class PassComputeStageExtents : public Pass {
public:
  PassComputeStageExtents();

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) override;
};

} // namespace dawn

#endif
