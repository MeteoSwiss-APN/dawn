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

#include "dawn/AST/ASTExpr.h"
#include "dawn/IIR/Extents.h"
#include "dawn/IIR/FieldAccessExtents.h"
#include "dawn/IIR/Interval.h"
#include "dawn/Support/Json.h"
#include <optional>
#include <utility>

namespace dawn {
namespace iir {
class StencilInstantiation;

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
                      ///  from where the Field has been accessed

  bool unstrucutred_;
  ast::Expr::LocationType location_ = ast::Expr::LocationType::Cells;

public:
  Field(Field&& f) = default;
  Field(Field const& f) = default;
  Field& operator=(Field&&) = default;
  Field& operator=(const Field&) = default;

  Field(int accessID, IntendKind intend, std::optional<Extents> const& readExtents,
        std::optional<Extents> const& writeExtents, Interval const& interval)
      : accessID_(accessID), intend_(intend), extents_(readExtents, writeExtents),
        extentsRB_(readExtents, writeExtents), interval_(interval), unstrucutred_(false) {}

  Field(int accessID, IntendKind intend, std::optional<Extents> const& readExtents,
        std::optional<Extents> const& writeExtents, Interval const& interval,
        ast::Expr::LocationType location)
      : accessID_(accessID), intend_(intend), extents_(readExtents, writeExtents),
        extentsRB_(readExtents, writeExtents), interval_(interval), unstrucutred_(true),
        location_(location) {}

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

  std::optional<Extents> const& getReadExtents() const { return extents_.getReadExtents(); }
  std::optional<Extents> const& getWriteExtents() const { return extents_.getWriteExtents(); }

  std::optional<Extents> const& getReadExtentsRB() const { return extentsRB_.getReadExtents(); }
  std::optional<Extents> const& getWriteExtentsRB() const { return extentsRB_.getWriteExtents(); }

  json::json jsonDump() const;

  Extents const& getExtents() const { return extents_.getExtents(); }
  Extents const& getExtentsRB() const { return extentsRB_.getExtents(); }

  IntendKind getIntend() const { return intend_; }
  int getAccessID() const { return accessID_; }
  /// @}

  /// @brief setters
  /// @{
  void setIntend(IntendKind intend) { intend_ = intend; }
  void setReadExtentsRB(Extents const& extents) { extentsRB_.setReadExtents(extents); }
  void setWriteExtentsRB(Extents const& extents) { extentsRB_.setWriteExtents(extents); }
  void setReadExtentsRB(std::optional<Extents> const& extents);
  void setWriteExtentsRB(std::optional<Extents> const& extents);
  /// @}

  // Enclosing interval where accesses where recorded,
  /// i.e. interval_.extend(Extent)
  Interval computeAccessedInterval() const;

  /// @brief merge of the extents
  /// @{
  void mergeReadExtents(Extents const& extents) { extents_.mergeReadExtents(extents); }
  void mergeWriteExtents(Extents const& extents) { extents_.mergeWriteExtents(extents); }
  void mergeReadExtents(std::optional<Extents> const& extents) {
    extents_.mergeReadExtents(extents);
  }
  void mergeWriteExtents(std::optional<Extents> const& extents) {
    extents_.mergeWriteExtents(extents);
  }

  void mergeReadExtentsRB(Extents const& extents) { extentsRB_.mergeReadExtents(extents); }
  void mergeWriteExtentsRB(Extents const& extents) { extentsRB_.mergeWriteExtents(extents); }

  void mergeReadExtentsRB(std::optional<Extents> const& extents) {
    extentsRB_.mergeReadExtents(extents);
  }
  void mergeWriteExtentsRB(std::optional<Extents> const& extents) {
    extentsRB_.mergeWriteExtents(extents);
  }
  /// @}
  ///
  void extendInterval(Interval const& interval) { interval_.merge(interval); }

  ast::Expr::LocationType getLocation() { return location_; }

  bool isUnstructured() { return unstrucutred_; }
};

/// @brief merges all the fields from sourceFields into destinationFields
/// If a baseExtent is provided (optionally), the extent of each sourceField is expanded with the
/// baseExtent (in order to account for redundant block computations where the accesses were
/// recorded)
void mergeFields(std::unordered_map<int, Field> const& sourceFields,
                 std::unordered_map<int, Field>& destinationFields,
                 std::optional<Extents> baseExtents = std::optional<Extents>());

void mergeField(const Field& sField, Field& dField);

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
