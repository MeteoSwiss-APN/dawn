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

#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/Options.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Serialization/SIRSerializer.h"

#include <list>
#include <map>
#include <memory>
#include <string>

namespace dawn {

/// @brief List of default optimizer pass groups
std::list<PassGroup> defaultPassGroups();

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

} // namespace dawn
