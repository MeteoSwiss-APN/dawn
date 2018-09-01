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
namespace iir {

Interval Field::computeAccessedInterval() const {
  Interval accessedInterval = interval_;
  accessedInterval = accessedInterval.extendInterval(getExtents());
  return accessedInterval;
}

void mergeFields(std::unordered_map<int, Field> const& sFields,
                 std::unordered_map<int, Field>& dFields, boost::optional<Extents> baseExtents) {

  boost::optional<Extents> baseHoriExtents = baseExtents;
  if(baseExtents.is_initialized()) {
    (*baseHoriExtents)[2].Minus = 0;
    (*baseHoriExtents)[2].Plus = 0;
  }
  for(const auto& fieldPair : sFields) {
    Field sField = fieldPair.second;

    auto readExtentsRB = sField.getReadExtents();
    if(readExtentsRB.is_initialized() && baseHoriExtents.is_initialized()) {
      readExtentsRB->expand(*baseHoriExtents);
      sField.setReadExtentsRB(readExtentsRB);
    }

    auto writeExtentsRB = sField.getWriteExtents();
    if(writeExtentsRB.is_initialized() && baseHoriExtents.is_initialized()) {
      writeExtentsRB->expand(*baseHoriExtents);
      sField.setWriteExtentsRB(writeExtentsRB);
    }

    auto it = dFields.find(sField.getAccessID());
    if(it != dFields.end()) {
      // Adjust the Intend
      if(it->second.getIntend() == Field::IK_Input && sField.getIntend() == Field::IK_Output)
        it->second.setIntend(Field::IK_InputOutput);
      else if(it->second.getIntend() == Field::IK_Output && sField.getIntend() == Field::IK_Input)
        it->second.setIntend(Field::IK_InputOutput);

      // field accounting for extents of the accesses plus the base extent (i.e. normally redundant
      // computations of the stages)

      // Merge the Extent
      it->second.mergeReadExtentsRB(sField.getReadExtentsRB());
      it->second.mergeWriteExtentsRB(sField.getWriteExtentsRB());

      it->second.mergeReadExtents(sField.getReadExtents());
      it->second.mergeWriteExtents(sField.getWriteExtents());
      it->second.extendInterval(sField.getInterval());
    } else {

      // add the baseExtent of the field (i.e. normally redundant computations of a stage)
      dFields.emplace(sField.getAccessID(), sField);
    }
  }
}
} // namespace iir
} // namespace dawn
