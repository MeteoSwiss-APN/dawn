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

#include "dawn/Optimizer/PassManager.h"
#include "dawn/Support/DiagnosticsEngine.h"
#include "dawn/Support/NonCopyable.h"
#include <map>
#include <memory>

namespace dawn {

struct SIR;
namespace sir {
struct Stencil;
}
namespace iir {
class StencilInstantiation;
}

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
public:
  struct OptimizerContextOptions {
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  TYPE NAME = DEFAULT_VALUE;
#include "dawn/Optimizer/OptimizerOptions.inc"
#undef OPT
  };

private:
  DiagnosticsEngine& diagnostics_;
  OptimizerContextOptions options_;

  const std::shared_ptr<SIR> SIR_;
  std::map<std::string, std::shared_ptr<iir::StencilInstantiation>> stencilInstantiationMap_;
  PassManager passManager_;
  HardwareConfig hardwareConfiguration_;

  void fillIIR();

public:
  /// @brief Initialize the context with a SIR
  OptimizerContext(DiagnosticsEngine& diagnostics, OptimizerContextOptions options,
                   const std::shared_ptr<SIR>& SIR);

  /// @brief Initialize the context with a stencil instantiation map
  OptimizerContext(DiagnosticsEngine& diagnostics, OptimizerContextOptions options,
                   std::map<std::string, std::shared_ptr<iir::StencilInstantiation>> const&
                       stencilInstantiationMap);

  /// @brief Get StencilInstantiation map
  std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>& getStencilInstantiationMap();
  const std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>&
  getStencilInstantiationMap() const;

  /// @brief Check if there are errors
  bool hasErrors() const { return getDiagnostics().hasErrors(); }

  /// @brief Get the PassManager
  PassManager& getPassManager() { return passManager_; }
  const PassManager& getPassManager() const { return passManager_; }

  /// @brief Get the SIR
  const std::shared_ptr<SIR>& getSIR() const { return SIR_; }

  /// @brief Get options
  const OptimizerContextOptions& getOptions() const;
  OptimizerContextOptions& getOptions();

  /// @brief Get the diagnostics engine
  const DiagnosticsEngine& getDiagnostics() const;
  DiagnosticsEngine& getDiagnostics();

  /// @brief Get the hardware configuration
  const HardwareConfig& getHardwareConfiguration() const { return hardwareConfiguration_; }
  HardwareConfig& getHardwareConfiguration() { return hardwareConfiguration_; }

  /// @brief Create a new pass at the end of the pass list
  template <class T, typename... Args>
  void pushBackPass(Args&&... args) {
    std::unique_ptr<T> pass = std::make_unique<T>(*this, std::forward<Args>(args)...);
    passManager_.getPasses().push_back(std::move(pass));
  }

  /// @brief fillIIRFromSIR
  /// @param stencilInstantation
  /// @param SIRStencil
  /// @param fullSIR
  /// @return
  bool fillIIRFromSIR(std::shared_ptr<iir::StencilInstantiation> stencilInstantation,
                      const std::shared_ptr<sir::Stencil> SIRStencil,
                      const std::shared_ptr<SIR> fullSIR);
  bool restoreIIR(std::string const& name,
                  std::shared_ptr<iir::StencilInstantiation> stencilInstantiation);
};

} // namespace dawn

#endif
