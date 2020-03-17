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

#ifndef DAWN_OPTIMIZER_PASSSETLOOPORDER_H
#define DAWN_OPTIMIZER_PASSSETLOOPORDER_H

#include "dawn/Optimizer/Pass.h"

namespace dawn {

/// @brief This Pass sets the loop order of multistages to parallel if possible
class PassSetLoopOrder : public Pass {
public:
  PassSetLoopOrder(OptimizerContext& context);

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) override;
};

} // namespace dawn

#endif // DAWN_OPTIMIZER_PASSSETLOOPORDER_H
