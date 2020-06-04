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

#pragma once

#include "dawn/CodeGen/Options.h"
#include "dawn/CodeGen/TranslationUnit.h"
#include "dawn/Optimizer/Driver.h"
#include "dawn/Optimizer/Options.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/SIRSerializer.h"

namespace dawn {

/// @brief Convenience function to compile SIR directly to a translation unit
std::unique_ptr<codegen::TranslationUnit> compile(
    const std::shared_ptr<SIR>& stencilIR, const std::list<PassGroup>& groups = defaultPassGroups(),
    const Options& optimizerOptions = {}, codegen::Backend backend = codegen::Backend::GridTools,
    const codegen::Options& codegenOptions = {});

/// @brief Convenience function to compile SIR directly to a translation unit. Use strings in place
/// of C++ structures.
std::string compile(const std::string& sir, SIRSerializer::Format format,
                    const std::list<PassGroup>& groups, const Options& optimizerOptions,
                    codegen::Backend backend, const codegen::Options& codegenOptions);

} // namespace dawn
