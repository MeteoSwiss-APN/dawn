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

#ifndef GSL_COMPILER_GSLCOMPILER_H
#define GSL_COMPILER_GSLCOMPILER_H

#include "gsl/CodeGen/TranslationUnit.h"
#include "gsl/Compiler/DiagnosticsEngine.h"
#include "gsl/Compiler/Options.h"
#include "gsl/Support/NonCopyable.h"
#include <memory>

namespace gsl {

struct SIR;

/// @brief The GSLCompiler class
/// @ingroup compiler
class GSLCompiler : NonCopyable {
  std::unique_ptr<DiagnosticsEngine> diagnostics_;
  std::unique_ptr<Options> options_;
  std::string filename_;

public:
  /// @brief Code generation backend
  enum CodeGenKind { CG_GTClang, CG_GTClangNaiveCXX };

  /// @brief Initialize the compiler by setting up diagnostics
  GSLCompiler(Options* options = nullptr);

  /// @brief Compile the SIR using the provided code generation routine
  /// @returns compiled TranslationUnit on success, `nullptr` otherwise
  std::unique_ptr<TranslationUnit> compile(const SIR* SIR, CodeGenKind codeGen);

  /// @brief Get options
  const Options& getOptions() const;
  Options& getOptions();

  /// @brief Get the diagnostics engine
  const DiagnosticsEngine& getDiagnostics() const;
  DiagnosticsEngine& getDiagnostics();
};

} // namespace gsl

#endif
