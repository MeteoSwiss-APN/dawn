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

#ifndef DAWN_COMPILER_DRIVER_H
#define DAWN_COMPILER_DRIVER_H

#include "dawn/CodeGen/Driver.h"
#include "dawn/CodeGen/Options.h"
#include "dawn/CodeGen/TranslationUnit.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerOptions.h"
#include <list>
#include <map>
#include <string>

namespace dawn {

// TODO This will be moved to Compiler/Driver.h when
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
  DataLocalityMetric
};

/// @brief List of default optimizer pass groups.
std::list<PassGroup> defaultPassGroups();

/// @brief Lower to IIR and run groups
std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>
run(const std::shared_ptr<SIR>& stencilIR, const std::list<PassGroup>& groups,
    const OptimizerOptions& options = {});

/// @brief Run groups
std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>
run(const std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>&
        stencilInstantiationMap,
    const std::list<PassGroup>& groups, const OptimizerOptions& options = {});

/// @brief Compile SIR using default pass groups
std::unique_ptr<codegen::TranslationUnit> compile(const std::shared_ptr<SIR>& stencilIR,
                                                  codegen::Backend backend,
                                                  const OptimizerOptions& optimizerOptions = {},
                                                  const codegen::Options& codegenOptions = {});

} // namespace dawn

#endif
