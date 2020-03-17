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

#include <list>
#include <memory>

namespace dawn {

struct SIR;

/// @brief Enumeration of all pass groups
enum class PassGroup {
  Parallel,
  SSA,
  PrintStencilGraph,
  SetStageName,
  StageReordering,
  StageMerger,
  TemporaryMerger,
  Inlining,
  IntervalPartitioning,
  TmpToStencilFunction,
  SetNonTempCaches,
  SetCaches,
  SetBlockSize,
  DataLocalityMetric,
  SetLoopOrder,
};

/// @brief The DawnCompiler class
/// @ingroup compiler
class DawnCompiler : NonCopyable {
  DiagnosticsEngine diagnostics_;
  Options options_;

public:
  /// @brief Initialize the compiler by setting up diagnostics
  DawnCompiler() = default;
  DawnCompiler(const Options& options);

  /// @brief Apply parallelizer, code optimization, and generate
  std::unique_ptr<codegen::TranslationUnit> compile(const std::shared_ptr<SIR>& stencilIR,
                                                    std::list<PassGroup> groups = {});

  /// @brief Lower to IIRs
  std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>
  lowerToIIR(const std::shared_ptr<SIR>& stencilIR);

  /// @brief Run optimization passes on the IIRs
  std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>
  optimize(const std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>&
               stencilInstantiationMap,
           const std::list<PassGroup>& groups);

  /// @brief Generate a translation unit from a set of Stencil Instantiations
  std::unique_ptr<codegen::TranslationUnit>
  generate(const std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>&
               stencilInstantiationMap);

  static std::list<PassGroup> defaultPassGroups();

  /// @brief Get options
  const Options& getOptions() const;
  Options& getOptions();

  /// @brief Get the diagnostics engine
  const DiagnosticsEngine& getDiagnostics() const;
  DiagnosticsEngine& getDiagnostics();
};

} // namespace dawn

#endif
