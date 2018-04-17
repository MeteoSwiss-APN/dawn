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

#ifndef DAWN_OPTIMIZER_INTERVAL_H
#define DAWN_OPTIMIZER_INTERVAL_H

#include "dawn/Optimizer/Extents.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/HashCombine.h"
#include <algorithm>
#include <iterator>
#include <unordered_set>
#include <vector>

namespace dawn {

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
  int lowerLevel_;
  int upperLevel_;
  int lowerOffset_;
  int upperOffset_;

public:
  enum class Bound { upper = 0, lower };

  /// @name Constructors and Assignment
  /// @{
  Interval(int lowerLevel, int upperLevel, int lowerOffset = 0, int upperOffset = 0)
      : lowerLevel_(lowerLevel), upperLevel_(upperLevel), lowerOffset_(lowerOffset),
        upperOffset_(upperOffset) {}

  Interval(const sir::Interval& interval)
      : lowerLevel_(interval.LowerLevel), upperLevel_(interval.UpperLevel),
        lowerOffset_(interval.LowerOffset), upperOffset_(interval.UpperOffset) {}

  Interval(const Interval&) = default;
  Interval(Interval&&) = default;
  Interval& operator=(const Interval&) = default;
  Interval& operator=(Interval&&) = default;
  /// @}

  int lowerLevel() const { return lowerLevel_; }
  int upperLevel() const { return upperLevel_; }
  int lowerOffset() const { return lowerOffset_; }
  int upperOffset() const { return upperOffset_; }

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
  inline int lowerBound() const { return (lowerLevel_ + lowerOffset_); }

  /// @brief Get the upper bound of the Interval (i.e `upperLevel + upperOffset`)
  inline int upperBound() const { return (upperLevel_ + upperOffset_); }

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
  void merge(const Interval& other) {
    int lb = lowerBound(), ub = upperBound();
    lowerLevel_ = std::min(lowerLevel_, other.lowerLevel());
    upperLevel_ = std::max(upperLevel_, other.upperLevel());
    lowerOffset_ = lb < other.lowerBound() ? lb - lowerLevel_ : other.lowerBound() - lowerLevel();
    upperOffset_ = ub > other.upperBound() ? ub - upperLevel_ : other.upperBound() - upperLevel();
  }

  /// @brief Create a @b new interval which is extented by the given Extents
  ///
  /// If extents is `{0, 0, 0, 0, -1, 1}` (i.e the vertical extent is `<-1, 1>` we create a new
  /// interval (from `this`) which has a decreased lowerOffset by -1 and an increased upper
  /// Offset by +1.
  Interval extendInterval(const Extent& verticalExtent) const;

  Interval extendInterval(const Extents& extents) const { return extendInterval(extents[2]); }

  /// @brief Convert to SIR Interval
  sir::Interval asSIRInterval() const {
    return sir::Interval(lowerLevel_, upperLevel_, lowerOffset_, upperOffset_);
  }

  /// @brief returns true if the level bound of the interval is the end of the axis
  bool levelIsEnd(Bound bound) const {
    return (bound == Bound::lower) ? lowerLevelIsEnd() : upperLevelIsEnd();
  }

  size_t overEnd() const { return (upperLevelIsEnd() && (upperOffset_ > 0)) ? upperOffset_ : 0; }

  size_t belowBegin() const {
    return (!lowerLevelIsEnd() && (lowerBound() < 0)) ? (size_t)-lowerOffset_ : 0;
  }

  /// @brief returns true if the lower bound of the interval is the end of the axis
  bool lowerLevelIsEnd() const { return (lowerLevel_ == sir::Interval::End); }

  /// @brief returns true if the upper bound of the interval is the end of the axis
  bool upperLevelIsEnd() const { return (upperLevel_ == sir::Interval::End); }

  /// @brief Convert interval to string
  std::string toString() const;

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

} // namespace dawn

namespace std {

template <>
struct hash<dawn::Interval> {
  size_t operator()(const dawn::Interval& I) const {
    std::size_t seed = 0;
    dawn::hash_combine(seed, I.lowerLevel() + I.lowerOffset(), I.upperLevel() + I.upperOffset());
    return seed;
  }
};

} // namespace std

#endif
