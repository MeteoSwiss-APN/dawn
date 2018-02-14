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

#ifndef DAWN_OPTIMIZER_OPTIMIZERCONTEXT_H
#define DAWN_OPTIMIZER_OPTIMIZERCONTEXT_H

#include "dawn/Compiler/DiagnosticsEngine.h"
#include "dawn/Compiler/Options.h"
#include "dawn/Optimizer/PassManager.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/Support/NonCopyable.h"
#include <map>
#include <memory>

namespace dawn {

struct SIR;
class StencilInstantiation;
class DawnCompiler;

struct HardwareConfig {
  /// Maximum number of fields concurrently in shared memory
  int SMemMaxFields = 8;

  /// Maximum number of fields concurrently in the texture cache
  int TexCacheMaxFields = 3;
};

/// @brief Context of handling all Optimizations
/// @ingroup optimizer
class OptimizerContext : NonCopyable {

  DiagnosticsEngine& diagnostics_;
  Options& options_;

  const std::shared_ptr<SIR> SIR_;
  std::map<std::string, std::shared_ptr<StencilInstantiation>> stencilInstantiationMap_;
  PassManager passManager_;
  HardwareConfig hardwareConfiguration_;

public:
  /// @brief Initialize the context with a SIR
  OptimizerContext(DiagnosticsEngine& diagnostics, Options& options,
                   const std::shared_ptr<SIR>& SIR);

  /// @brief Get StencilInstantiation map
  std::map<std::string, std::shared_ptr<StencilInstantiation>>& getStencilInstantiationMap();
  const std::map<std::string, std::shared_ptr<StencilInstantiation>>&
  getStencilInstantiationMap() const;

  /// @brief Check if there are errors
  bool hasErrors() const { return getDiagnostics().hasErrors(); }

  /// @brief Get the PassManager
  PassManager& getPassManager() { return passManager_; }
  const PassManager& getPassManager() const { return passManager_; }

  /// @brief Get the SIR
  const std::shared_ptr<SIR> getSIR() const { return SIR_; }

  /// @brief Get options
  const Options& getOptions() const;
  Options& getOptions();

  /// @brief Get the diagnostics engine
  const DiagnosticsEngine& getDiagnostics() const;
  DiagnosticsEngine& getDiagnostics();

  /// @brief Get the hardware configuration
  const HardwareConfig& getHardwareConfiguration() const { return hardwareConfiguration_; }
  HardwareConfig& getHardwareConfiguration() { return hardwareConfiguration_; }
};

} // namespace dawn

#endif
