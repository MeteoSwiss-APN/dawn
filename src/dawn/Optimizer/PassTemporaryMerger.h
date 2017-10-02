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
class PassTemporaryMerger : public Pass {
public:
  PassTemporaryMerger();

  /// @brief Pass implementation
  bool run(StencilInstantiation* stencilInstantiation) override;
};

} // namespace dawn

#endif
