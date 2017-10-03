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

#include "dawn/Optimizer/Interval.h"
#include <iostream>
#include <set>
#include <sstream>

namespace dawn {

std::string Interval::toString() const {
  std::stringstream ss;
  ss << *this;
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const Interval& interval) {
  auto printLevel = [&](int level, int offset) -> void {
    if(level == sir::Interval::Start)
      os << "Start";
    else if(level == sir::Interval::End)
      os << "End";
    else
      os << level;

    if(offset != 0)
      os << (offset > 0 ? "+" : "") << offset;
  };

  os << "{ ";
  printLevel(interval.lowerLevel(), interval.lowerOffset());
  os << " : ";
  printLevel(interval.upperLevel(), interval.upperOffset());
  os << " }";

  return os;
}

std::string Interval::makeCodeGenName(const Interval& interval) {
  std::stringstream ss;
  ss << "interval";

  auto printLevelAndOffset = [&](int level, int offset) -> void {
    ss << "_";
    if(level == sir::Interval::Start)
      ss << "start";
    else if(level == sir::Interval::End)
      ss << "end";
    else
      ss << level;
    ss << "_";
    if(offset != 0)
      ss << (offset > 0 ? "plus_" : "minus_");
    ss << std::abs(offset);
  };

  printLevelAndOffset(interval.lowerLevel(), interval.lowerOffset());
  printLevelAndOffset(interval.upperLevel(), interval.upperOffset());
  return ss.str();
}

std::vector<Interval> Interval::computeLevelUnion(const std::vector<Interval>& intervals) {
  std::set<int> levels;
  for(const Interval& interval : intervals) {
    levels.insert(interval.lowerLevel());
    levels.insert(interval.upperLevel());
  }

  std::vector<Interval> newIntervals;

  int lowerLevel = *levels.begin();
  newIntervals.emplace_back(lowerLevel, lowerLevel);
  for(auto it = std::next(levels.begin()), end = levels.end(); it != end; ++it) {
    int upperLevel = *it;

    if(((upperLevel - 1) - (lowerLevel + 1)) >= 0)
      newIntervals.emplace_back(lowerLevel + 1, upperLevel - 1);

    newIntervals.emplace_back(upperLevel, upperLevel);
    lowerLevel = upperLevel;
  }

  return newIntervals;
}

std::vector<Interval> Interval::computeGapIntervals(const Interval& axis,
                                                    const std::vector<Interval>& intervals) {
  std::vector<Interval> newIntervals;

  // Insert the intervals in sorted order
  for(const Interval& interval : intervals) {
    auto it = std::find_if(newIntervals.begin(), newIntervals.end(), [&](const Interval& I) {
      DAWN_ASSERT_MSG(I.lowerBound() != interval.lowerBound(),
                      "Intervals have to be non-overlapping");
      return I.lowerBound() > interval.lowerBound();
    });
    newIntervals.insert(it, interval);
  }

  // Close the intermediate gaps
  if(newIntervals.size() > 1) {
    for(auto curLowIt = newIntervals.begin(), curTopIt = std::next(newIntervals.begin());
        curTopIt != newIntervals.end();) {
      const Interval& curLowInterval = *curLowIt;
      const Interval& curTopInterval = *curTopIt;

      DAWN_ASSERT_MSG(!curLowInterval.overlaps(curTopInterval),
                      "Intervals have to be non-overlapping");

      if(!curLowInterval.adjacent(curTopInterval)) {
        Interval gapFillInterval(curLowInterval.upperLevel(), curTopInterval.lowerLevel(),
                                 curLowInterval.upperOffset() + 1,
                                 curTopInterval.lowerOffset() - 1);
        curLowIt = newIntervals.insert(curTopIt, gapFillInterval);
        curTopIt = std::next(curLowIt);
      }

      curLowIt = curTopIt;
      curTopIt++;
    }
  }

  // Close the lower gap
  if(axis.lowerBound() != newIntervals.front().lowerBound()) {
    const Interval& lowestInterval = newIntervals.front();
    Interval lowerFillInterval(axis.lowerLevel(), lowestInterval.lowerLevel(), axis.lowerOffset(),
                               lowestInterval.lowerOffset() - 1);
    newIntervals.insert(newIntervals.begin(), lowerFillInterval);
  }

  // Close the upper gap
  if(axis.upperBound() != newIntervals.back().upperBound()) {
    const Interval& topInterval = newIntervals.back();
    Interval topFillInterval(topInterval.upperLevel(), axis.upperLevel(),
                             topInterval.upperOffset() + 1, axis.upperOffset());
    newIntervals.insert(newIntervals.end(), topFillInterval);
  }

  return newIntervals;
}

} // namespace dawn
