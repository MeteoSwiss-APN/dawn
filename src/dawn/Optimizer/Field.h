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

#ifndef DAWN_OPTIMIZER_FIELDTYPES_H
#define DAWN_OPTIMIZER_FIELDTYPES_H

#include "dawn/Optimizer/Extents.h"
#include "dawn/Optimizer/Interval.h"
#include <utility>

namespace dawn {

/// @brief Information of a field
///
/// Fields are sorted primarily according on their `Intend` and secondarily on their `AccessID`.
/// Fields are hashed on their `AccessID`.
///
/// @ingroup optimizer
struct Field {
  enum IntendKind { IK_Output = 0, IK_InputOutput = 1, IK_Input = 2 };

  int AccessID;       ///< Unique AccessID of the field
  IntendKind Intend;  ///< Intended usage
  Extents Extent;     ///< Accumulated extent of the field
  Interval interval_; ///< Enclosing Interval from the iteration space
                      ///  where the Field has been accessed

  Field(int accessID, IntendKind intend, Interval interval)
      : AccessID(accessID), Intend(intend), Extent{}, interval_(interval) {}

  /// @name Operators
  /// @{
  bool operator==(const Field& other) const {
    return (AccessID == other.AccessID && Intend == other.Intend);
  }
  bool operator!=(const Field& other) const { return !(*this == other); }
  bool operator<(const Field& other) const {
    return (Intend < other.Intend || (Intend == other.Intend && AccessID < other.AccessID));
  }
  /// @}

  /// @brief get the interval where field has been accessed
  Interval getAccessedInterval() const { return Interval(interval_).extendInterval(Extent); }
};

} // namespace dawn

namespace std {

template <>
struct hash<dawn::Field> {
  size_t operator()(const dawn::Field& field) const { return std::hash<int>()(field.AccessID); }
};

} // namespace std

#endif
