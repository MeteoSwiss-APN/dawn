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

#include "dawn/IIR/Field.h"

namespace dawn {
// namespace iir {

void mergeFields(std::unordered_map<int, Field> const& sFields,
                 std::unordered_map<int, Field>& dFields) {
  for(const auto& fieldPair : sFields) {
    const Field& field = fieldPair.second;
    auto it = dFields.find(field.getAccessID());
    if(it != dFields.end()) {
      // Adjust the Intend
      if(it->second.getIntend() == Field::IK_Input && field.getIntend() == Field::IK_Output)
        it->second.setIntend(Field::IK_InputOutput);
      else if(it->second.getIntend() == Field::IK_Output && field.getIntend() == Field::IK_Input)
        it->second.setIntend(Field::IK_InputOutput);

      // Merge the Extent
      it->second.mergeReadExtents(field.getReadExtents());
      it->second.mergeWriteExtents(field.getWriteExtents());
      it->second.extendInterval(field.getInterval());
    } else
      dFields.emplace(field.getAccessID(), field);
  }
}

//} // namespace iir
} // namespace dawn
