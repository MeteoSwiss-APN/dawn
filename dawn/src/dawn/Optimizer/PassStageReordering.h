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
#include "dawn/Optimizer/ReorderStrategy.h"

namespace dawn {

/// @brief Pass for reordering stages to increase data locality
///
/// This Pass depends on `PassSetStageGraph`.
///
/// @ingroup optimizer
///
/// This pass is not necessary to create legal code and is hence not in the debug-group
class PassStageReordering : public Pass {
public:
  PassStageReordering(ReorderStrategy::Kind strategy)
      : Pass("PassStageReordering"), strategy_(strategy) {
    dependencies_.push_back("PassSetStageGraph");
  }

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
           const Options& options = {}) override;

private:
  ReorderStrategy::Kind strategy_;
};

} // namespace dawn
