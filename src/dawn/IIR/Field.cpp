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

void mergeFields(std::unordered_map<int, Field> const& sourceFields,
                 std::unordered_map<int, Field>& destinationFields,
                 boost::optional<Extents> baseExtents) {

  for(const auto& fieldPair : sourceFields) {
    Field sField = fieldPair.second;

    auto readExtentsRB = sField.getReadExtents();
    if(readExtentsRB.is_initialized() && baseExtents.is_initialized()) {
      readExtentsRB->expand(*baseExtents);
      sField.setReadExtentsRB(readExtentsRB);
    }

    auto writeExtentsRB = sField.getWriteExtents();
    if(writeExtentsRB.is_initialized() && baseExtents.is_initialized()) {
      writeExtentsRB->expand(*baseExtents);
      sField.setWriteExtentsRB(writeExtentsRB);
    }

    auto it = destinationFields.find(sField.getAccessID());
    if(it != destinationFields.end()) {
      // Adjust the Intend
      if(it->second.getIntend() != sField.getIntend())
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
      destinationFields.emplace(sField.getAccessID(), sField);
    }
  }
}

void Field::setReadExtentsRB(boost::optional<Extents> const& extents) {
  if(extents.is_initialized()) {
    extentsRB_.setReadExtents(*extents);
  }
}
void Field::setWriteExtentsRB(boost::optional<Extents> const& extents) {
  if(extents.is_initialized()) {
    extentsRB_.setWriteExtents(*extents);
  }
}

} // namespace iir
} // namespace dawn
