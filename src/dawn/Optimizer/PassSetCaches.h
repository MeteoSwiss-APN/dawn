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

#include "dawn/Optimizer/Interval.h"
#include "dawn/Optimizer/Pass.h"
#include "dawn/Optimizer/Stage.h"

namespace dawn {

/// @brief Determine which fields can be cached during the executation of the multi-stage
///
/// @ingroup optimizer
///
/// This pass is not necessary to create legal code and is hence not in the debug-group
class PassSetCaches : public Pass {
public:
  PassSetCaches();

  /// @brief Pass implementation
  bool run(const std::shared_ptr<StencilInstantiation>& stencilInstantiation) override;
};

} // namespace dawn

#endif
