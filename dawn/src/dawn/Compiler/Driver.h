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

#include "dawn/CodeGen/Driver.h"
#include "dawn/CodeGen/Options.h"
#include "dawn/CodeGen/TranslationUnit.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/Driver.h"
#include "dawn/Optimizer/Options.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Serialization/SIRSerializer.h"

#include <list>
#include <map>
#include <memory>
#include <string>

namespace dawn {

/// TODO Move these to Optimizer/Driver when OptimizerContext is removed
/// {
/// @brief Lower to IIR and run groups
std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>
run(const std::shared_ptr<SIR>& stencilIR, const std::list<PassGroup>& groups,
    const Options& options = {});

/// @brief Lower to IIR and run groups. Use strings in place of C++ structures.
///
/// NOTE: This method always returns the stencil instantiations as json string objects, not
/// bytes, as this greatly simplifies the conversion.
/// See https://pybind11.readthedocs.io/en/stable/advanced/cast/strings.html for more details.
std::map<std::string, std::string> run(const std::string& sir, dawn::SIRSerializer::Format format,
                                       const std::list<dawn::PassGroup>& groups,
                                       const dawn::Options& options = {});

/// @brief Run groups
std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>
run(const std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>&
        stencilInstantiationMap,
    const std::list<PassGroup>& groups, const Options& options = {});

/// @brief Run groups. Use strings in place of C++ structures.
///
/// NOTE: This method always returns the stencil instantiations as json string objects, not bytes,
/// as this greatly simplifies the conversion.
/// See https://pybind11.readthedocs.io/en/stable/advanced/cast/strings.html for more details.
std::map<std::string, std::string>
run(const std::map<std::string, std::string>& stencilInstantiationMap, IIRSerializer::Format format,
    const std::list<PassGroup>& groups, const Options& options = {});
/// }

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
