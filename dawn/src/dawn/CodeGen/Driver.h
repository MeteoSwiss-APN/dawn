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
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Serialization/IIRSerializer.h"

#include <map>
#include <memory>
#include <string>

namespace dawn {
namespace codegen {

/// @brief Parse the backend string to enumeration
Backend parseBackendString(const std::string& backendStr);

/// @brief Run the code generation
std::unique_ptr<TranslationUnit>
run(const std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>& context,
    Backend backend, const Options& options = {});

/// @brief Run the code generation. Use strings instead of C++ objects.
std::string run(const std::map<std::string, std::string>& stencilInstantiationMap,
                IIRSerializer::Format format, codegen::Backend backend,
                const codegen::Options& options = {});

/// @brief Run the code generation for a single stencil instantiation.
std::unique_ptr<TranslationUnit>
run(const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation, Backend backend,
    const Options& options = {});

/// @brief Shortcut to generate code from a translation unit
std::string generate(const std::unique_ptr<TranslationUnit>& translationUnit);

} // namespace codegen
} // namespace dawn
