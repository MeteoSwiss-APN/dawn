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
#include "dawn/IIR/StencilMetaInformation.h"

namespace dawn {
namespace iir {

Interval Field::computeAccessedInterval() const {
  Interval accessedInterval = interval_;
  accessedInterval = accessedInterval.extendInterval(getExtents());
  return accessedInterval;
}

json::json Field::jsonDump() const {
  json::json node;
  node["accessID"] = accessID_;
  node["intend"] = static_cast<int>(intend_);
  node["extents"] = extents_.jsonDump();
  node["redundant extents"] = extentsRB_.jsonDump();
  std::stringstream ss;
  ss << interval_;
  node["interval"] = ss.str();

  node["dim"] = fieldDimensions_.toString();

  return node;
}

void mergeField(const Field& sField, Field& dField) {

  // Adjust the Intend
  if(dField.getIntend() != sField.getIntend())
    dField.setIntend(Field::IntendKind::InputOutput);

  // field accounting for extents of the accesses plus the base extent (i.e. normally redundant
  // computations of the stages)

  // Merge the Extent
  dField.mergeReadExtentsRB(sField.getReadExtentsRB());
  dField.mergeWriteExtentsRB(sField.getWriteExtentsRB());

  dField.mergeReadExtents(sField.getReadExtents());
  dField.mergeWriteExtents(sField.getWriteExtents());
  dField.extendInterval(sField.getInterval());
}

void mergeFields(std::unordered_map<int, Field> const& sourceFields,
                 std::unordered_map<int, Field>& destinationFields,
                 std::optional<Extents> baseExtents) {

  for(const auto& fieldPair : sourceFields) {
    Field sField = fieldPair.second;

    auto readExtentsRB = sField.getReadExtents();
    if(readExtentsRB && baseExtents) {
      *readExtentsRB += *baseExtents;
      sField.setReadExtentsRB(readExtentsRB);
    }

    auto writeExtentsRB = sField.getWriteExtents();
    if(writeExtentsRB && baseExtents) {
      *writeExtentsRB += *baseExtents;
      sField.setWriteExtentsRB(writeExtentsRB);
    }

    auto it = destinationFields.find(sField.getAccessID());
    if(it != destinationFields.end()) {
      mergeField(sField, it->second);
    } else {

      // add the baseExtent of the field (i.e. normally redundant computations of a stage)
      destinationFields.emplace(sField.getAccessID(), sField);
    }
  }
}

void Field::setReadExtentsRB(std::optional<Extents> const& extents) {
  if(extents) {
    extentsRB_.setReadExtents(*extents);
  }
}
void Field::setWriteExtentsRB(std::optional<Extents> const& extents) {
  if(extents) {
    extentsRB_.setWriteExtents(*extents);
  }
}

} // namespace iir
} // namespace dawn
