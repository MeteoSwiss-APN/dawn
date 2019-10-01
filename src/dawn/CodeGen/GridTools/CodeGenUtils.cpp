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

#include "dawn/CodeGen/GridTools/CodeGenUtils.h"
#include "dawn/Support/IndexRange.h"
#include <sstream>

namespace dawn {
namespace codegen {
namespace gt {

std::vector<std::string>
CodeGenUtils::buildPlaceholderList(const iir::StencilMetaInformation& metadata,
                                   const std::map<int, iir::Stencil::FieldInfo>& stencilFields,
                                   const sir::GlobalVariableMap& globalsMap, bool buildPair) {
  auto nonTempFields = makeRange(
      stencilFields,
      std::function<bool(std::pair<int, iir::Stencil::FieldInfo> const&)>(
          [](std::pair<int, iir::Stencil::FieldInfo> const& p) { return !p.second.IsTemporary; }));

  std::vector<std::string> placeholders;
  for(const auto& fieldInfoPair : nonTempFields) {
    const auto& fieldName = fieldInfoPair.second.Name;

    std::stringstream placeholderStatement;
    placeholderStatement << "p_" + fieldName;
    if(buildPair) {
      placeholderStatement << "{} = ";
      // TODO(havogt): refactor
      // metadata.isAccessType(iir::FieldAccessType::FAT_InterStencilTemporary, fieldInfoPair.first)
      // to
      // metadata.isInterStencilTemporary(fieldInfoPair.first)
      if(metadata.isAccessType(iir::FieldAccessType::FAT_InterStencilTemporary,
                               fieldInfoPair.first)) {
        placeholderStatement << "m_" << fieldName;
      } else {
        placeholderStatement << fieldName;
      }
    }
    placeholders.push_back(placeholderStatement.str());
  }
  return placeholders;
}

} // namespace gt
} // namespace codegen
} // namespace dawn
