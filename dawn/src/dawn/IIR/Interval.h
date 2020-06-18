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

#pragma once

#include "dawn/IIR/Extents.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/HashCombine.h"
#include <algorithm>
#include <iterator>
#include <unordered_set>
#include <vector>

namespace dawn {
namespace iir {

/// @brief Representation of a vertical interval, given by a lower and upper bound where a bound
/// is represented by a level and an offset (`bound = level + offset`)
///
/// The Interval has to satisfy the following invariants:
///  - `lowerLevel >= sir::Interval::Start`
///  - `upperLevel <= sir::Interval::End`
///  - `(lowerLevel + lowerOffset) <= (upperLevel + upperOffset)`
///
/// @see sir::Interval
/// @ingroup optimizer
class Interval {
public:
  struct IntervalLevel {
    int levelMark_;
    int offset_;
    int bound() const { return (levelMark_ + offset_); }
    bool isEnd() const { return (levelMark_ == sir::Interval::End); }
  };

private:
  IntervalLevel lower_, upper_;

public:
  enum class Bound { upper = 0, lower };

  /// @name Constructors and Assignment
  /// @{
  Interval(int lowerLevel, int upperLevel, int lowerOffset = 0, int upperOffset = 0)
      : lower_{lowerLevel, lowerOffset}, upper_{upperLevel, upperOffset} {}

  Interval(const sir::Interval& interval)
      : lower_{interval.LowerLevel, interval.LowerOffset}, upper_{interval.UpperLevel,
                                                                  interval.UpperOffset} {}

  Interval() = delete;
  Interval(const Interval&) = default;
  Interval(Interval&&) = default;
  Interval& operator=(const Interval&) = default;
  Interval& operator=(Interval&&) = default;
  /// @}

  int lowerLevel() const { return lower_.levelMark_; }
  int upperLevel() const { return upper_.levelMark_; }
  int lowerOffset() const { return lower_.offset_; }
  int upperOffset() const { return upper_.offset_; }

  void invert();

  /// @brief the Interval allows to construct mathematically invalid intervals where the lower bound
  /// is higher than the upper bound. This method allows to check if the interval is mathematically
  /// well defined
  inline bool valid() { return (lowerBound() <= upperBound()); }

  IntervalLevel upperIntervalLevel() const { return upper_; }
  IntervalLevel lowerIntervalLevel() const { return lower_; }

  /// @brief computes the intersection of two intervals
  Interval intersect(const Interval& other) const;

  /// @brief crop the interval around the window of one of the bounds of the interval
  /// (notice that the window can specify offsets so that the cropped interval can also extend and
  /// go beyod the limits of the
  /// original window)
  Interval crop(Bound bound, std::array<int, 2> window) const;

  int offset(const Bound bound) const {
    return (bound == Bound::lower) ? lowerOffset() : upperOffset();
  }
  int level(const Bound bound) const {
    return (bound == Bound::lower) ? lowerLevel() : upperLevel();
  }

  /// @brief Get the bound of the Interval (i.e `level + offset`)
  inline int bound(const Bound bound) const {
    return (bound == Bound::lower) ? lowerBound() : upperBound();
  }

  /// @brief Get the lower bound of the Interval (i.e `lowerLevel + lowerOffset`)
  inline int lowerBound() const { return lower_.bound(); }

  /// @brief Get the upper bound of the Interval (i.e `upperLevel + upperOffset`)
  inline int upperBound() const { return upper_.bound(); }

  /// @name Comparison operator
  /// @{
  bool operator==(const Interval& other) const {
    return lowerBound() == other.lowerBound() && upperBound() == other.upperBound();
  }
  bool operator!=(const Interval& other) const { return !(*this == other); }
  /// @}

  /// @brief Check if `this` overlaps with `other`
  ///
  /// We check that there is an integer `C` which is in `[lowerBound(), upperBound()]` @b and
  /// `[other.lowerBound(), other.upperBound()]`. Note that we have the invariant
  /// `LowerBound <= UpperBound`
  bool overlaps(const Interval& other) const {
    return lowerBound() <= other.upperBound() && other.lowerBound() <= upperBound();
  }

  /// @brief Check if `this` fully contains `other`
  bool contains(const Interval& other) const {
    return other.lowerBound() >= lowerBound() && other.upperBound() <= upperBound();
  }

  /// @brief Check if `this` is adjacent to `other`
  bool adjacent(const Interval& other) const {
    if(overlaps(other))
      return false;

    if(upperBound() < other.lowerBound())
      return (other.lowerBound() - upperBound() == 1);
    else // other.upperBound() < lowerBound()
      return (lowerBound() - other.upperBound() == 1);
  }

  /// @brief Merge `this` with `other` and assign an Interval to `this` which is the union of the
  /// two
  void merge(const Interval& other);

  /// @brief Create a @b new interval which is extented by the given Extents
  ///
  /// If extents is `{0, 0, 0, 0, -1, 1}` (i.e the vertical extent is `<-1, 1>` we create a new
  /// interval (from `this`) which has a decreased lowerOffset by -1 and an increased upper
  /// Offset by +1.
  Interval extendInterval(const Extent& verticalExtent) const;

  inline Interval extendInterval(const Extents& extents) const {
    return extendInterval(extents.verticalExtent());
  }

  /// @brief Convert to SIR Interval
  inline sir::Interval asSIRInterval() const {
    return sir::Interval(lowerLevel(), upperLevel(), lowerOffset(), upperOffset());
  }

  /// @brief returns true if the level bound of the interval is the end of the axis
  inline bool levelIsEnd(Bound bound) const {
    return (bound == Bound::lower) ? lowerLevelIsEnd() : upperLevelIsEnd();
  }

  inline size_t overEnd() const {
    return (upperLevelIsEnd() && (upper_.offset_ > 0)) ? upper_.offset_ : 0;
  }

  inline size_t belowBegin() const {
    return (!lowerLevelIsEnd() && (lowerBound() < 0)) ? (size_t)-lower_.offset_ : 0;
  }

  /// @brief returns true if the lower bound of the interval is the end of the axis
  bool lowerLevelIsEnd() const { return lower_.isEnd(); }

  /// @brief returns true if the upper bound of the interval is the end of the axis
  bool upperLevelIsEnd() const { return upper_.isEnd(); }

  /// @brief Convert interval to string
  explicit operator std::string() const;
  std::string toString() const;
  std::string toStringGen() const;

  friend std::ostream& operator<<(std::ostream& os, const Interval& interval);

  /// @brief Generate a name uniquely identifying the interval for code generation
  ///
  /// @ingroup optimizer
  static std::string makeCodeGenName(const Interval& interval);

  /// @brief Compute a contigeous, non-overlapping set of intervals spanning the entire `axis` by
  /// filling the gaps of `intervals`
  ///
  /// The `intervals` do not need to be sorted but they can't be overlapping. The computed
  /// intervals are sorted in ascending order (i.e the first intervals is the lowest).
  ///
  /// @b Example:
  /// Given the axis A = [0, 10] (given by a lower and upper bound) and 2 intervals (I1, I2) with:
  ///   - I1 = [0, 1]
  ///   - I2 = [3, 5]
  ///
  /// The computed gap intverals would be G1 and G2 with:
  ///   - G1 = [2, 2]
  ///   - G2 = [6, 10]
  ///
  /// and this function will return (R1, R2, R3, R4):
  ///   - R1 = [0, 1]
  ///   - R2 = [2, 2]
  ///   - R3 = [3, 5]
  ///   - R4 = [6, 10]
  ///
  /// Hence, we filled the "gaps".
  ///
  /// @ingroup optimizer
  static std::vector<Interval> computeGapIntervals(const Interval& axis,
                                                   const std::vector<Interval>& intervals);

  ///
  /// @brief computes a partition of the set of levels contained in the collection of intervals
  /// such that the number of subsets of the partition is minimized given the constraint that
  /// the intersection of any pair of interval (if not null) should be split as a subset.
  /// @b Example:
  /// @code
  ///   computePartition(vector<int>(Interval(2,5), Interval(4,7)))
  /// @endcode
  /// generates the output
  /// @code
  ///   vector<int>(Interval(2,3), Interval(4,5), Interval(6,7))
  /// @endcode
  /// @ingroup optimizer
  ///
  static std::vector<Interval> computePartition(const std::vector<Interval>& intervals);

  /// @brief Computes a set of non-overlapping, adjacent intervals of the given set of intervals
  /// where all interval levels are preserved (i.e a union of all levels of the given intervals)
  ///
  /// Each level is represented as a single interval.
  ///
  /// @b Example:
  /// Given 4 intervals (I1, I2, I3, I4) with:
  ///   - I1 = [0, 20]
  ///   - I2 = [5, 21]
  ///   - I3 = [5, 10]
  ///   - I4 = [21, 21]
  ///
  /// The computed union would span the axis [0, 21] with levels at {5, 10, 20} i.e in this
  /// case the function would return 8 intervals:
  ///   - CI1 = [0, 0]
  ///   - CI2 = [1, 4]
  ///   - CI3 = [5, 5]
  ///   - CI4 = [6, 9]
  ///   - CI5 = [10, 10]
  ///   - CI6 = [11, 19]
  ///   - CI7 = [20, 20]
  ///   - CI8 = [21, 21]
  ///
  /// @ingroup optimizer
  static std::vector<Interval> computeLevelUnion(const std::vector<Interval>& intervals);
};

/// @brief struct that contains an interval and a associated generated name
/// Should be used for data structures where the interval name generation (expensive string
/// operation) is accessed frequently
struct IntervalProperties {
  /// @name Constructors and Assignment
  /// @{

  IntervalProperties(Interval const& interval)
      : interval_(interval), name_(Interval::makeCodeGenName(interval)) {}

  IntervalProperties(const IntervalProperties&) = default;
  IntervalProperties(IntervalProperties&&) = default;
  IntervalProperties& operator=(const IntervalProperties&) = default;
  IntervalProperties& operator=(IntervalProperties&&) = default;
  /// @}

  /// @name Comparison operator
  /// @{
  bool operator==(const IntervalProperties& other) const { return interval_ == other.interval_; }
  bool operator!=(const IntervalProperties& other) const { return !(*this == other); }
  /// @}

  Interval interval_;
  std::string name_;
};

struct IntervalDiff {
  enum class RangeType { literal, fullRange, minusFullRange };
  RangeType rangeType_;
  int value;

  bool null() const {
    if(rangeType_ != RangeType::literal)
      return false;
    return (value == 0);
  }
};

IntervalDiff distance(Interval::IntervalLevel f, Interval::IntervalLevel s);
IntervalDiff distance(Interval f, Interval s, LoopOrderKind order);

IntervalDiff operator+(IntervalDiff idiff, int val);
bool operator==(const IntervalDiff& first, const IntervalDiff& second);

Interval::IntervalLevel advance(Interval::IntervalLevel& lev, LoopOrderKind loopOrder, int step);

} // namespace iir
} // namespace dawn

namespace std {

template <>
struct hash<dawn::iir::Interval> {
  size_t operator()(const dawn::iir::Interval& I) const {
    std::size_t seed = 0;
    dawn::hash_combine(seed, I.lowerLevel() + I.lowerOffset(), I.upperLevel() + I.upperOffset());
    return seed;
  }
};

template <>
struct hash<dawn::iir::IntervalProperties> {
  size_t operator()(const dawn::iir::IntervalProperties& I) const {
    return hash<dawn::iir::Interval>()(I.interval_);
  }
};

} // namespace std
