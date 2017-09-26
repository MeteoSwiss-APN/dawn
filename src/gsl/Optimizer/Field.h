//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#ifndef GSL_OPTIMIZER_FIELDTYPES_H
#define GSL_OPTIMIZER_FIELDTYPES_H

#include "gsl/Optimizer/Extents.h"
#include <utility>

namespace gsl {

/// @brief Information of a field
///
/// Fields are sorted primarily according on their `Intend` and secondarily on their `AccessID`.
/// Fields are hashed on their `AccessID`.
///
/// @ingroup optimizer
struct Field {
  enum IntendKind { IK_Output = 0, IK_InputOutput = 1, IK_Input = 2 };

  int AccessID;      ///< Unique AccessID of the field
  IntendKind Intend; ///< Intended usage
  Extents Extent;    ///< Accumulated extent of the field

  Field(int accessID, IntendKind intend) : AccessID(accessID), Intend(intend), Extent{} {}

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
};

} // namespace gsl

namespace std {

template <>
struct hash<gsl::Field> {
  size_t operator()(const gsl::Field& field) const { return std::hash<int>()(field.AccessID); }
};

} // namespace std

#endif
