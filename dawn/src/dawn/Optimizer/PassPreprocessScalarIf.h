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

/// @brief ...
/// * Input:  
/// * Output: 
/// @ingroup optimizer
///
/// This pass is necessary to generate legal IIR
class PassPreprocessScalarIf : public Pass {
public:
  PassPreprocessScalarIf(OptimizerContext& context) : Pass(context, "PassPreprocessScalarIf") {}

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) override;
};

} // namespace dawn
