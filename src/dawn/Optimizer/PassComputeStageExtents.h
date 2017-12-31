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

#include "dawn/Optimizer/Pass.h"

namespace dawn {

/// @brief This Pass computes and assign the stage graph of each stencil
///
/// This Pass depends on `PassSetStageName`.
///
/// @ingroup optimizer
class PassComputeStageExtents : public Pass {
public:
  PassComputeStageExtents();

  /// @brief Pass implementation
  bool run(StencilInstantiation* stencilInstantiation) override;
};

} // namespace dawn

#endif
