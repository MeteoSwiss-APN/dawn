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

#ifndef DAWN_OPTIMIZER_PASSINTEGRITYCHECK_H
#define DAWN_OPTIMIZER_PASSINTEGRITYCHECK_H

#include "dawn/Optimizer/Pass.h"
#include "dawn/Validator/IntegrityChecker.h"
#include "dawn/Validator/GridTypeChecker.h"
#include "dawn/Validator/LocationTypeChecker.h"

namespace dawn {

/// @brief Perform basic integrity checks
///
/// @ingroup optimizer
///
/// This pass is read-only and is hence not in the debug-group
class PassValidation : public Pass {
public:
  PassValidation(OptimizerContext &context);

  /// @brief Pass run implementation for IIR
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) override;

  /// @brief SIR validation
  bool run(const std::shared_ptr<dawn::SIR>& sir);
};

} // namespace dawn

#endif // DAWN_OPTIMIZER_PASSINTEGRITYCHECK_H
