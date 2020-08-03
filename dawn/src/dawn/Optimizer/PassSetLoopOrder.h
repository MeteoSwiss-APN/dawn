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

namespace dawn {

/// @brief This Pass sets the loop order of multistages to parallel if possible
class PassSetLoopOrder : public Pass {
public:
  PassSetLoopOrder(): Pass("PassSetLoopOrder") {}

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
           const Options& options = {}) override;
};

} // namespace dawn
