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

#ifndef DAWN_IIR_FIELD_H
#define DAWN_IIR_FIELD_H

#include "dawn/IIR/Extents.h"
#include "dawn/IIR/FieldAccessExtents.h"
#include "dawn/IIR/Interval.h"
#include <utility>

namespace dawn {
namespace iir {

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
  int accessID_;               ///< Unique AccessID of the field
  IntendKind intend_;          ///< Intended usage
  FieldAccessExtents extents_; ///< Accumulated read and write extent of the field
  FieldAccessExtents
      extentsRB_; ///< Accumulated read and write extent of the field, extended by the
  /// redundant computation of a block
  Interval interval_; ///< Enclosing Interval from the iteration space
                      ///  where the Field has been accessed
public:
  Field(Field&& f) = default;
  Field(Field const& f) = default;
  Field& operator=(Field&&) = default;
  Field& operator=(const Field&) = default;

  Field(int accessID, IntendKind intend, boost::optional<Extents> const& readExtents,
        boost::optional<Extents> const& writeExtents, Interval const& interval)
      : accessID_(accessID), intend_(intend),
        extents_(FieldAccessExtents(readExtents, writeExtents)),
        extentsRB_(FieldAccessExtents(readExtents, writeExtents)), interval_(interval) {}

  Field(int accessID, IntendKind intend, boost::optional<Extents>&& readExtents,
        boost::optional<Extents>&& writeExtents, Interval&& interval)
      : accessID_(accessID), intend_(intend),
        extents_(FieldAccessExtents(std::move(readExtents), std::move(writeExtents))),
        extentsRB_(extents_), interval_(std::move(interval)) {}

  /// @name Operators
  /// @{
  inline bool operator==(const Field& other) const {
    return (accessID_ == other.accessID_ && intend_ == other.intend_);
  }
  inline bool operator!=(const Field& other) const { return !(*this == other); }
  inline bool operator<(const Field& other) const {
    return (intend_ < other.intend_ || (intend_ == other.intend_ && accessID_ < other.accessID_));
  }
  /// @}

  /// @brief getters
  /// @{
  inline Interval const& getInterval() const { return interval_; }

  // Enclosing interval where accesses where recorded,
  /// i.e. interval_.extend(Extent)
  Interval computeAccessedInterval() const;

  inline boost::optional<Extents> const& getReadExtents() const {
    return extents_.getReadExtents();
  }
  inline boost::optional<Extents> const& getWriteExtents() const {
    return extents_.getWriteExtents();
  }

  inline boost::optional<Extents> const& getReadExtentsRB() const {
    return extentsRB_.getReadExtents();
  }
  inline boost::optional<Extents> const& getWriteExtentsRB() const {
    return extentsRB_.getWriteExtents();
  }

  inline Extents const& getExtents() const { return extents_.getExtents(); }
  inline Extents const& getExtentsRB() const { return extentsRB_.getExtents(); }

  inline IntendKind getIntend() const { return intend_; }
  inline int getAccessID() const { return accessID_; }
  /// @}

  /// @brief setters
  /// @{
  inline void setIntend(IntendKind intend) { intend_ = intend; }
  /// @}

  inline void mergeReadExtents(Extents const& extents) { extents_.mergeReadExtents(extents); }
  inline void mergeWriteExtents(Extents const& extents) { extents_.mergeWriteExtents(extents); }
  inline void mergeReadExtents(boost::optional<Extents> const& extents) {
    extents_.mergeReadExtents(extents);
  }
  inline void mergeWriteExtents(boost::optional<Extents> const& extents) {
    extents_.mergeWriteExtents(extents);
  }

  inline void setReadExtentsRB(Extents const& extents) { extentsRB_.setReadExtents(extents); }
  inline void setWriteExtentsRB(Extents const& extents) { extentsRB_.setWriteExtents(extents); }
  inline void setReadExtentsRB(boost::optional<Extents> const& extents) {
    if(extents.is_initialized()) {
      extentsRB_.setReadExtents(*extents);
    }
  }
  inline void setWriteExtentsRB(boost::optional<Extents> const& extents) {
    if(extents.is_initialized()) {
      extentsRB_.setWriteExtents(*extents);
    }
  }

  inline void mergeReadExtentsRB(Extents const& extents) { extentsRB_.mergeReadExtents(extents); }
  inline void mergeWriteExtentsRB(Extents const& extents) { extentsRB_.mergeWriteExtents(extents); }

  inline void mergeReadExtentsRB(boost::optional<Extents> const& extents) {
    extentsRB_.mergeReadExtents(extents);
  }
  inline void mergeWriteExtentsRB(boost::optional<Extents> const& extents) {
    extentsRB_.mergeWriteExtents(extents);
  }
  //  inline void expandReadExtents(Extents const& extents) { extents_.expandReadExtents(extents); }
  //  inline void expandWriteExtents(Extents const& extents) { extents_.expandWriteExtents(extents);
  //  }
  //  inline void expandReadExtents(boost::optional<Extents> const& extents) {
  //    extents_.expandReadExtents(extents);
  //  }
  //  inline void expandWriteExtents(boost::optional<Extents> const& extents) {
  //    extents_.expandWriteExtents(extents);
  //  }
  inline void extendInterval(Interval const& interval) { interval_.merge(interval); }
};

void mergeFields(std::unordered_map<int, Field> const& sourceFields,
                 std::unordered_map<int, Field>& destinationFields,
                 boost::optional<Extents> baseExtents = boost::optional<Extents>());

} // namespace iir
} // namespace dawn

namespace std {

template <>
struct hash<dawn::iir::Field> {
  size_t operator()(const dawn::iir::Field& field) const {
    return std::hash<int>()(field.getAccessID());
  }
};

} // namespace std

#endif
