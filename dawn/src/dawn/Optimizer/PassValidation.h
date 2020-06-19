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
#include "dawn/SIR/SIR.h"

namespace dawn {

/// @brief Perform basic integrity checks
///
/// @ingroup optimizer
///
/// This pass is read-only and is hence not in the debug-group
class PassValidation : public Pass {
public:
  PassValidation(OptimizerContext& context);

  /// @brief Pass run implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& instantiation) override;

  /// @brief IIR validation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& instantiation,
           const std::string& description);

  /// @brief SIR validation
  bool run(const std::shared_ptr<dawn::SIR>& sir);
};

} // namespace dawn
