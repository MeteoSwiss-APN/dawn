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

#include "dawn/IIR/Stencil.h"
#include <map>
#include <string>
#include <vector>

namespace dawn {
namespace codegen {
namespace gt {

struct CodeGenUtils {
  // build the collection of placeholder typedef names
  static std::vector<std::string>
  buildPlaceholderList(const iir::StencilMetaInformation& metadata,
                       const std::map<int, iir::Stencil::FieldInfo>& stencilFields,
                       const sir::GlobalVariableMap& globalsMap, bool buildPair = false);
};

} // namespace gt
} // namespace codegen
} // namespace dawn
