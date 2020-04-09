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

#include "dawn/Compiler/Driver.h"
#include "dawn/CodeGen/TranslationUnit.h"
#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/Compiler/Options.h"

namespace dawn {

std::list<PassGroup> defaultPassGroups() {
  return {PassGroup::SetStageName, PassGroup::StageReordering, PassGroup::StageMerger,
          PassGroup::SetCaches, PassGroup::SetBlockSize};
}

std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>
run(const std::shared_ptr<SIR>& stencilIR, const std::list<PassGroup>& groups,
    const OptimizerOptions& options) {
  // Put all options there
  dawn::Options dawnOptions;
  // Copy over options passed in
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  dawnOptions.NAME = options.NAME;
#include "dawn/Optimizer/Options.inc"
#undef OPT
  DawnCompiler compiler(dawnOptions);
  return compiler.optimize(compiler.lowerToIIR(stencilIR), groups);
}

std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>
run(const std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>&
        stencilInstantiationMap,
    const std::list<PassGroup>& groups, const OptimizerOptions& options) {
  // Put all options there
  dawn::Options dawnOptions;
  // Copy over options passed in
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  dawnOptions.NAME = options.NAME;
#include "dawn/Optimizer/Options.inc"
#undef OPT
  DawnCompiler compiler(dawnOptions);
  return compiler.optimize(stencilInstantiationMap, groups);
}

std::unique_ptr<codegen::TranslationUnit> compile(const std::shared_ptr<SIR>& stencilIR,
                                                  const std::list<PassGroup>& passGroups,
                                                  const OptimizerOptions& optimizerOptions,
                                                  codegen::Backend backend,
                                                  const codegen::Options& codegenOptions) {
  return codegen::run(run(stencilIR, passGroups, optimizerOptions), backend, codegenOptions);
}

} // namespace dawn
