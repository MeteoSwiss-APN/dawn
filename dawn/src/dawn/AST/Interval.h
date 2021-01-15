#pragma once

#include "dawn/Support/Assert.h"
#include "dawn/Support/ComparisonHelpers.h"

namespace dawn {
namespace ast {

//===------------------------------------------------------------------------------------------===//
//     Interval
//===------------------------------------------------------------------------------------------===//

/// @brief Representation of a vertical interval, given by a lower and upper bound where a bound
/// is represented by a level and an offset (`bound = level + offset`)
///
/// The Interval has to satisfy the following invariants:
///  - `lowerLevel >= Interval::Start`
///  - `upperLevel <= Interval::End`
///  - `(lowerLevel + lowerOffset) <= (upperLevel + upperOffset)`
///
/// @ingroup sir
struct Interval {
  enum LevelKind : int { Start = 0, End = (1 << 20) };

  Interval(int lowerLevel, int upperLevel, int lowerOffset = 0, int upperOffset = 0)
      : LowerLevel(lowerLevel), UpperLevel(upperLevel), LowerOffset(lowerOffset),
        UpperOffset(upperOffset) {
    DAWN_ASSERT(lowerLevel >= LevelKind::Start && upperLevel <= LevelKind::End);
    DAWN_ASSERT((lowerLevel + lowerOffset) <= (upperLevel + upperOffset));
  }

  int LowerLevel;
  int UpperLevel;
  int LowerOffset;
  int UpperOffset;

  /// @name Comparison operator
  /// @{
  bool operator==(const Interval& other) const {
    return LowerLevel == other.LowerLevel && UpperLevel == other.UpperLevel &&
           LowerOffset == other.LowerOffset && UpperOffset == other.UpperOffset;
  }
  bool operator!=(const Interval& other) const { return !(*this == other); }

  CompareResult comparison(const Interval& rhs) const;
  /// @}

  /// @brief Convert to string
  /// @{
  std::string toString() const;
  friend std::ostream& operator<<(std::ostream& os, const Interval& interval);
  /// @}
};

} // namespace sir
} // namespace dawn