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

#ifndef DAWN_OPTIMIZER_PASSNONTEMPCACHES_H
#define DAWN_OPTIMIZER_PASSNONTEMPCACHES_H

#include "dawn/Optimizer/Pass.h"

namespace dawn {

/// @brief This Pass optimizes data locality on a per-multistage basis
///
/// For all the non-temporary, horizontally non-pointwise but vertically pointwise fields that we
/// access in any given multistage, we check how high the performance gain of using a local ij-cache
/// woud be.
///
/// If is is beneficial, we cache the fields with the most performance gain up to HardwareConfig's
/// Maximum number of fields concurrently in shared memory (defined in PassDataLocalityMetric.h)
///
/// The caching invokes a renaming of all the occurances of any given field we cache and adds
/// potentially filler stages and definitely flush stages to the multistage thus changing the ISIR
/// (StencilInstantation) in memory.
///
/// This Pass has no dependecies
///
/// @ingroup optimizer
///
/// This pass is not necessary to create legal code and is hence not in the debug-group
class PassSetNonTempCaches : public Pass {
public:
  PassSetNonTempCaches();

  /// @brief Pass implementation
  bool run(const std::shared_ptr<StencilInstantiation>& stencilInstantiation) override;
};

} // namespace dawn

#endif
