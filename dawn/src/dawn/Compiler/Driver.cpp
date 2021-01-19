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
#include "dawn/CodeGen/Driver.h"

namespace dawn {

std::unique_ptr<codegen::TranslationUnit> compile(const std::shared_ptr<SIR>& stencilIR,
                                                  const std::list<PassGroup>& passGroups,
                                                  const Options& optimizerOptions,
                                                  codegen::Backend backend,
                                                  const codegen::Options& codegenOptions) {
  return codegen::run(run(stencilIR, passGroups, optimizerOptions), backend, codegenOptions);
}

std::string compile(const std::string& sir, SIRSerializer::Format format,
                    const std::list<PassGroup>& groups, const Options& optimizerOptions,
                    codegen::Backend backend, const codegen::Options& codegenOptions) {
  auto stencilIR = SIRSerializer::deserializeFromString(sir, format);
  return codegen::generate(compile(stencilIR, groups, optimizerOptions, backend, codegenOptions));
}

} // namespace dawn
