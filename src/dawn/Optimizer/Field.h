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
class Field {
public:
  enum IntendKind { IK_Output = 0, IK_InputOutput = 1, IK_Input = 2 };

private:
  int accessID_;              ///< Unique AccessID of the field
  IntendKind intend_;         ///< Intended usage
  Extents extents_;           ///< Accumulated extent of the field
  Interval interval_;         ///< Enclosing Interval from the iteration space
                              ///  where the Field has been accessed
  Interval accessedInterval_; ///< Enclosing interval where accesses where recorded,
                              /// i.e. interval_.extend(Extent)

  void updateAccessedInterval() {
    accessedInterval_ = interval_;
    accessedInterval_ = accessedInterval_.extendInterval(extents_);
  }

public:
  Field(int accessID, IntendKind intend, Extents extents, Interval interval)
      : accessID_(accessID), intend_(intend), extents_(extents), interval_(interval),
        accessedInterval_(interval) {
    updateAccessedInterval();
  }

  /// @name Operators
  /// @{
  bool operator==(const Field& other) const {
    return (accessID_ == other.accessID_ && intend_ == other.intend_);
  }
  bool operator!=(const Field& other) const { return !(*this == other); }
  bool operator<(const Field& other) const {
    return (intend_ < other.intend_ || (intend_ == other.intend_ && accessID_ < other.accessID_));
  }
  /// @}

  /// @brief getters
  /// @{
  Interval const& getInterval() const { return interval_; }
  Interval const& getAccessedInterval() const { return accessedInterval_; }
  Extents const& getExtents() const { return extents_; }
  IntendKind getIntend() const { return intend_; }
  int getAccessID() const { return accessID_; }
  /// @}

  /// @brief setters
  /// @{
  void setIntend(IntendKind intend) { intend_ = intend; }
  /// @}

  void mergeExtents(Extents const& extents) {
    extents_.merge(extents);
    updateAccessedInterval();
  }
  void extendInterval(Interval const& interval) {
    interval_.merge(interval);
    updateAccessedInterval();
  }
};

} // namespace dawn

namespace std {

template <>
struct hash<dawn::Field> {
  size_t operator()(const dawn::Field& field) const {
    return std::hash<int>()(field.getAccessID());
  }
};

} // namespace std

#endif
