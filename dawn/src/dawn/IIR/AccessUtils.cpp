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
#include "dawn/IIR/AccessUtils.h"

namespace dawn {

namespace AccessUtils {

void recordWriteAccess(std::unordered_map<int, iir::Field>& inputOutputFields,
                       std::unordered_map<int, iir::Field>& inputFields,
                       std::unordered_map<int, iir::Field>& outputFields, int AccessID,
                       const std::optional<iir::Extents>& writeExtents,
                       iir::Interval const& doMethodInterval) {
  // Field was recorded as `InputOutput`, state can't change ...
  if(inputOutputFields.count(AccessID)) {
    inputOutputFields.at(AccessID).extendInterval(doMethodInterval);
    return;
  }

  // Field was recorded as `Input`, change it's state to `InputOutput`
  if(inputFields.count(AccessID)) {
    iir::Field& preField = inputFields.at(AccessID);
    preField.extendInterval(doMethodInterval);
    preField.setIntend(iir::Field::IK_InputOutput);
    inputOutputFields.insert({AccessID, preField});
    inputFields.erase(AccessID);
    return;
  }

  // Field not yet present, record it as output
  if(outputFields.count(AccessID)) {
    outputFields.at(AccessID).extendInterval(doMethodInterval);
  } else {
    outputFields.emplace(AccessID,
                         iir::Field(AccessID, iir::Field::IK_Output, std::optional<iir::Extents>(),
                                    writeExtents, doMethodInterval));
  }
}

void recordReadAccess(std::unordered_map<int, iir::Field>& inputOutputFields,
                      std::unordered_map<int, iir::Field>& inputFields,
                      std::unordered_map<int, iir::Field>& outputFields, int AccessID,
                      std::optional<iir::Extents> const& readExtents,
                      const iir::Interval& doMethodInterval) {

  // Field was recorded as `InputOutput`, state can't change ...
  if(inputOutputFields.count(AccessID)) {
    inputOutputFields.at(AccessID).extendInterval(doMethodInterval);
    return;
  }

  // Field was recorded as `Output`, change it's state to `InputOutput`
  if(outputFields.count(AccessID)) {
    iir::Field& preField = outputFields.at(AccessID);
    preField.extendInterval(doMethodInterval);
    preField.setIntend(iir::Field::IK_InputOutput);
    inputOutputFields.insert({AccessID, preField});

    outputFields.erase(AccessID);
    return;
  }

  // Field not yet present, record it as input
  if(inputFields.count(AccessID)) {
    inputFields.at(AccessID).extendInterval(doMethodInterval);
  } else {
    inputFields.emplace(AccessID, iir::Field(AccessID, iir::Field::IK_Input, readExtents,
                                             std::optional<iir::Extents>(), doMethodInterval));
  }
}
} // namespace AccessUtils
} // namespace dawn
