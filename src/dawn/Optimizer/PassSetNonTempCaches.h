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

#ifndef DAWN_OPTIMIZER_PASSNONTEMPCACHES_H
#define DAWN_OPTIMIZER_PASSNONTEMPCACHES_H

#include "dawn/Optimizer/Pass.h"

namespace dawn {

/// @brief This Pass prints the dependency graph of each stencil to a dot file
///
/// This Pass depends on `PassStageSplitter` (which sets the dependency graphs).
///
/// @ingroup optimizer
class PassSetNonTempCaches : public Pass {
public:
  PassSetNonTempCaches();

  /// @brief Pass implementation
  bool run(StencilInstantiation* stencilInstantiation) override;
};

} // namespace dawn

#endif
