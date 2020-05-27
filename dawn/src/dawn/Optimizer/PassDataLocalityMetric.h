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

private:
  static const int TERMINAL_CHAR_WIDTH = 70;

public:
  PassDataLocalityMetric(OptimizerContext& context);

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) override;
};

std::pair<int, int>
computeReadWriteAccessesMetric(const std::shared_ptr<iir::StencilInstantiation>& instantiation,
                               OptimizerContext& context, const iir::MultiStage& multiStage);
std::unordered_map<int, ReadWriteAccumulator> computeReadWriteAccessesMetricPerAccessID(
    const std::shared_ptr<iir::StencilInstantiation>& instantiation, OptimizerContext& context,
    const iir::MultiStage& multiStage);

} // namespace dawn
