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

#ifndef DAWN_COMPILER_DAWNCOMPILER_H
#define DAWN_COMPILER_DAWNCOMPILER_H

#include "dawn/CodeGen/TranslationUnit.h"
#include "dawn/Compiler/Options.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Support/DiagnosticsEngine.h"
#include "dawn/Support/NonCopyable.h"
#include <memory>

namespace dawn {

struct SIR;

/// @brief The DawnCompiler class
/// @ingroup compiler
class DawnCompiler : NonCopyable {
  DiagnosticsEngine diagnostics_;
  Options options_;

public:
  /// @brief Initialize the compiler by setting up diagnostics
  DawnCompiler() = default;
  DawnCompiler(const Options& options);

  /// @brief Compile the SIR using the provided code generation routine
  /// @returns compiled TranslationUnit on success, `nullptr` otherwise
  std::unique_ptr<codegen::TranslationUnit> compile(std::shared_ptr<SIR> const& SIR);

  std::unique_ptr<OptimizerContext> runOptimizer(std::shared_ptr<SIR> const& SIR);

  std::unique_ptr<codegen::TranslationUnit>
  generate(const std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>&
               stencilInstantiationMap);

  /// @brief Get options
  const Options& getOptions() const;
  Options& getOptions();

  /// @brief Get the diagnostics engine
  const DiagnosticsEngine& getDiagnostics() const;
  DiagnosticsEngine& getDiagnostics();
};

} // namespace dawn

#endif
