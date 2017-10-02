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

#ifndef DAWN_OPTIMIZER_PASSSETMULTISTAGECACHES_H
#define DAWN_OPTIMIZER_PASSSETMULTISTAGECACHES_H

#include "dawn/Optimizer/Pass.h"

namespace dawn {

/// @brief Determine which fields can be cached during the executation of the multi-stage
///
/// @ingroup optimizer
class PassSetCaches : public Pass {
public:
  PassSetCaches();

  /// @brief Pass implementation
  bool run(StencilInstantiation* stencilInstantiation) override;
};

} // namespace dawn

#endif
