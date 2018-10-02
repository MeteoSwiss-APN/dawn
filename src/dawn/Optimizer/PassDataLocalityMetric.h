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

#ifndef DAWN_OPTIMIZER_PASSDATALOCALITYMETRIC_H
#define DAWN_OPTIMIZER_PASSDATALOCALITYMETRIC_H

#include "dawn/IIR/MultiStage.h"
#include "dawn/Optimizer/Pass.h"

namespace dawn {

struct ReadWriteAccumulator {
  int numReads = 0;
  int numWrites = 0;

  int totalAccesses() const { return numReads + numWrites; }
};

/// @brief This Pass computes a heuristic measuring the data-locality of each stencil
///
/// @ingroup optimizer
///
/// This pass is not necessary to create legal code and is hence not in the debug-group
class PassDataLocalityMetric : public Pass {
public:
  PassDataLocalityMetric();

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) override;
};

std::pair<int, int>
computeReadWriteAccessesMetric(const iir::IIR* internalIR,
                               const iir::MultiStage& multiStage);
std::unordered_map<int, ReadWriteAccumulator> computeReadWriteAccessesMetricPerAccessID(const iir::IIR *internalIR,
    const iir::MultiStage& multiStage);

} // namespace dawn

#endif
