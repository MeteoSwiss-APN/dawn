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
#include "dawn/Compiler/DiagnosticsEngine.h"
#include "dawn/Compiler/Options.h"
#include "dawn/Support/NonCopyable.h"
#include <memory>

namespace dawn {

struct SIR;

/// @brief The DawnCompiler class
/// @ingroup compiler
class DawnCompiler : NonCopyable {
  std::unique_ptr<DiagnosticsEngine> diagnostics_;
  std::unique_ptr<Options> options_;
  std::string filename_;

public:
  /// @brief Code generation backend
  enum CodeGenKind { CG_GTClang = 0, CG_GTClangNaiveCXX, CG_GTClangOptCXX };

  /// @brief Initialize the compiler by setting up diagnostics
  DawnCompiler(Options* options = nullptr);

  /// @brief Compile the SIR using the provided code generation routine
  /// @returns compiled TranslationUnit on success, `nullptr` otherwise
  std::unique_ptr<codegen::TranslationUnit> compile(const SIR* SIR, CodeGenKind codeGen);

  /// @brief Get options
  const Options& getOptions() const;
  Options& getOptions();

  /// @brief Get the diagnostics engine
  const DiagnosticsEngine& getDiagnostics() const;
  DiagnosticsEngine& getDiagnostics();
};

} // namespace dawn

#endif
