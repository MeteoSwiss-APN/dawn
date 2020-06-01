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
#include "dawn/Optimizer/PassValidation.h"
#include "dawn/Support/NonCopyable.h"
#include "dawn/Support/STLExtras.h"
#include <list>
#include <memory>
#include <unordered_map>

namespace dawn {

namespace iir {
class StencilInstantiation;
}

/// @brief Handle registering and running of passes
class PassManager : public NonCopyable {
  std::list<std::unique_ptr<Pass>> passes_;
  std::unordered_map<std::string, int> passCounter_;

public:
  /// @brief Run all passes on the `instantiation`
  /// @returns `true` on success, `false` otherwise
  bool runAllPassesOnStencilInstantiation(
      OptimizerContext& context, const std::shared_ptr<iir::StencilInstantiation>& instantiation);

  /// @brief Run the given pass on the `instantiation`
  /// @returns `true` on success, `false` otherwise
  bool
  runPassOnStencilInstantiation(OptimizerContext& context,
                                const std::shared_ptr<iir::StencilInstantiation>& instantiation,
                                Pass* pass);

  /// @brief Get all registered passes
  std::list<std::unique_ptr<Pass>>& getPasses() { return passes_; }
  const std::list<std::unique_ptr<Pass>>& getPasses() const { return passes_; }
};

} // namespace dawn
