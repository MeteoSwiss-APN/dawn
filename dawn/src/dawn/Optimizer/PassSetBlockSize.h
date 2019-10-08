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

#ifndef DAWN_OPTIMIZER_PASSSETBLOCKSIZE_H
#define DAWN_OPTIMIZER_PASSSETBLOCKSIZE_H

#include "dawn/Optimizer/Pass.h"

namespace dawn {

/// @brief This Pass computes and assign the block size of each IIR
///
/// This Pass depends on `PassSetBlockSize`.
///
/// @ingroup optimizer
///
/// This pass is not necessary to create legal code and is hence not in the debug-group
class PassSetBlockSize : public Pass {
public:
  PassSetBlockSize(OptimizerContext& context);

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) override;
};

} // namespace dawn

#endif
