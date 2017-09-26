//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#ifndef GSL_OPTIMIZER_OPTIMIZERCONTEXT_H
#define GSL_OPTIMIZER_OPTIMIZERCONTEXT_H

#include "gsl/Compiler/DiagnosticsEngine.h"
#include "gsl/Compiler/Options.h"
#include "gsl/Optimizer/PassManager.h"
#include "gsl/Optimizer/StencilInstantiation.h"
#include "gsl/Support/NonCopyable.h"
#include <map>
#include <memory>

namespace gsl {

struct SIR;
class StencilInstantiation;
class GSLCompiler;

/// @brief Context of handling all Optimizations
/// @ingroup optimizer
class OptimizerContext : NonCopyable {
  GSLCompiler* compiler_;

  const SIR* SIR_;
  std::map<std::string, std::unique_ptr<StencilInstantiation>> stencilInstantiationMap_;
  PassManager passManager_;

public:
  /// @brief Initialize the context with a SIR
  OptimizerContext(GSLCompiler* compiler, const SIR* SIR);

  /// @brief Get StencilInstantiation map
  std::map<std::string, std::unique_ptr<StencilInstantiation>>& getStencilInstantiationMap();
  const std::map<std::string, std::unique_ptr<StencilInstantiation>>&
  getStencilInstantiationMap() const;

  /// @brief Check if there are errors
  bool hasErrors() const { return getDiagnostics().hasErrors(); }

  /// @brief Get the PassManager
  PassManager& getPassManager() { return passManager_; }
  const PassManager& getPassManager() const { return passManager_; }

  /// @brief Get the SIR
  const SIR* getSIR() const { return SIR_; }

  /// @brief Get options
  const Options& getOptions() const;
  Options& getOptions();

  /// @brief Get the diagnostics engine
  const DiagnosticsEngine& getDiagnostics() const;
  DiagnosticsEngine& getDiagnostics();
};

} // namespace gsl

#endif
