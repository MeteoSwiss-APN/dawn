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

#ifndef DAWN_OPTIMIZER_PASSSTAGEREORDERING_H
#define DAWN_OPTIMIZER_PASSSTAGEREORDERING_H

#include "dawn/Optimizer/Pass.h"
#include "dawn/Optimizer/ReorderStrategy.h"

namespace dawn {

/// @brief Pass for reordering stages to increase data locility
///
/// This Pass depends on `PassSetStageGraph`.
///
/// @ingroup optimizer
///
/// This pass is not necessary to create legal code and is hence not in the debug-group
class PassStageReordering : public Pass {
public:
  PassStageReordering(ReorderStrategy::ReorderStrategyKind strategy);

  /// @brief Pass implementation
  bool run(const std::shared_ptr<StencilInstantiation>& stencilInstantiation) override;

private:
  ReorderStrategy::ReorderStrategyKind strategy_;
};

} // namespace dawn

#endif
