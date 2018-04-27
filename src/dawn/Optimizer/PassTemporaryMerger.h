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

#ifndef DAWN_OPTIMIZER_PASSTEMPORARYMERGER_H
#define DAWN_OPTIMIZER_PASSTEMPORARYMERGER_H

#include "dawn/Optimizer/Pass.h"

namespace dawn {

/// @brief Pass to merge temporaries
/// @ingroup optimizer
///
/// This pass is not necessary to create legal code and is hence not in the debug-group
class PassTemporaryMerger : public Pass {
public:
  PassTemporaryMerger();

  /// @brief Pass implementation
  bool run(const std::shared_ptr<StencilInstantiation>& stencilInstantiation) override;
};

} // namespace dawn

#endif
