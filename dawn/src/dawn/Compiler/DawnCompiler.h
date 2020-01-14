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
  Options options_;
  DiagnosticsEngine diagnostics_;
  std::string filename_;

public:
  /// @brief Initialize the compiler by setting up diagnostics
  DawnCompiler();
  DawnCompiler(Options const& options);

  /// @brief Apply parallelizer, code optimization, and generate
  std::unique_ptr<codegen::TranslationUnit> compile(std::shared_ptr<SIR> const& stencilIR);

  /// @brief Apply parallelizer to stencilIR creating set of valid IIRs, one for each stencil
  std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>
  parallelize(std::shared_ptr<SIR> const& stencilIR);

  /// @brief Apply optimizer to a whole set of IIRs
  std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>
  optimize(std::map<std::string, std::shared_ptr<iir::StencilInstantiation>> const&
               stencilInstantiationMap);

  /// @brief Generate a translation unit from a set of Stencil Instantiations
  std::unique_ptr<codegen::TranslationUnit>
  generate(std::map<std::string, std::shared_ptr<iir::StencilInstantiation>> const&
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
