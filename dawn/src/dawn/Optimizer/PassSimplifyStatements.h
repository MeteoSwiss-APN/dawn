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

#include "Pass.h"

namespace dawn {

/// @brief PassSimplifyStatements...
/// @ingroup optimizer
///
/// This pass is necessary to generate legal IIR
class PassSimplifyStatements : public Pass {
public:
  PassSimplifyStatements(OptimizerContext& context) : Pass(context, "PassSimplifyStatements") {}

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) override;
};

} // namespace dawn
