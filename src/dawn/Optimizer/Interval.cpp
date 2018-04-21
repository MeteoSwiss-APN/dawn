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
#include <algorithm>
#include <iostream>
#include <set>
#include <sstream>
#include <unordered_set>

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

Interval Interval::extendInterval(const Extent& verticalExtent) const {
  return Interval(lowerLevel_, upperLevel_, lowerOffset_ + verticalExtent.Minus,
                  upperOffset_ + verticalExtent.Plus);
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

std::vector<Interval> Interval::computePartition(std::vector<Interval> const& intervals) {

  std::vector<Interval> newIntervals(intervals);

  // sort the intervals based on the lower bound
  std::sort(newIntervals.begin(), newIntervals.end(),
            [](Interval const& a, Interval const& b) { return a.lowerBound() < b.lowerBound(); });

  // When we have more than one interval, we sort them based on their lower bounds and evaluate
  // neighbouring pairs
  if(newIntervals.size() > 1) {
    int cnt = 0;
    bool change = false;
    // we start the loop from the interval with the smallest lower bound and iterate over them
    for(auto curLowIt = newIntervals.begin(), curTopIt = std::next(newIntervals.begin());
        curTopIt != newIntervals.end();) {
      // get iterators of two contiguous intervals
      const Interval& curLowInterval = *curLowIt;
      const Interval& curTopInterval = *curTopIt;

      // If they are the same, we simply eliminate one
      if(curLowInterval == curTopInterval) {
        curTopIt = newIntervals.erase(curTopIt);
        continue;
      }
      // if one interval is contained in the other hand, we split them in two non overlapping
      // intervals depending on how the contains are:
      // [a---------b]
      // [a--c]
      // ====>>
      // [a--c]
      //      [c+1--b]
      else if(curLowInterval.contains(curTopInterval) &&
              (curLowInterval.lowerBound() == curTopInterval.lowerBound())) {
        Interval splitHighInterval(curLowInterval.lowerLevel(), curTopInterval.upperLevel(),
                                   curLowInterval.lowerOffset(), curTopInterval.upperOffset());
        Interval splitLowInterval(curTopInterval.upperLevel(), curLowInterval.upperLevel(),
                                  curTopInterval.upperOffset() + 1, curLowInterval.upperOffset());

        *curLowIt = splitLowInterval;
        *curTopIt = splitHighInterval;

        change = true;
      }
      // [a----b]
      // [a-----------c]
      // ====>>
      // [a----b]
      //        [b+1--c]
      else if(curTopInterval.contains(curLowInterval)) {
        Interval splitLowInterval(curLowInterval.lowerLevel(), curLowInterval.upperLevel(),
                                  curLowInterval.lowerOffset(), curLowInterval.upperOffset());
        Interval splitHighInterval(curLowInterval.upperLevel(), curTopInterval.upperLevel(),
                                   curLowInterval.upperOffset() + 1, curTopInterval.upperOffset());

        *curLowIt = splitLowInterval;
        *curTopIt = splitHighInterval;

        change = true;
      }
      // [a------------b]
      //         [c----b]
      // ====>>
      // [a--c-1]
      //        [c-----b]
      else if(curLowInterval.contains(curTopInterval) &&
              (curLowInterval.upperBound() == curTopInterval.upperBound())) {
        Interval splitLowInterval(curLowInterval.lowerLevel(), curTopInterval.lowerLevel(),
                                  curLowInterval.lowerOffset(), curTopInterval.lowerOffset() - 1);
        Interval splitHighInterval(curTopInterval.lowerLevel(), curTopInterval.upperLevel(),
                                   curTopInterval.lowerOffset(), curTopInterval.upperOffset());

        *curLowIt = splitLowInterval;
        *curTopIt = splitHighInterval;

        change = true;
      }
      // [a----------------b]
      //   [c-------d]
      // ====>>
      // [a--c-1]
      //        [c--d]
      //             [d+1--b]
      // if the lower one competely covers the higher one, we need three intervals: the lower one,
      // the intersection and the top interval
      else if(curLowInterval.contains(curTopInterval)) {
        Interval splitLowInterval(curLowInterval.lowerLevel(), curTopInterval.lowerLevel(),
                                  curLowInterval.lowerOffset(), curTopInterval.lowerOffset() - 1);
        Interval splitMidInterval(curTopInterval.lowerLevel(), curTopInterval.upperLevel(),
                                  curTopInterval.lowerOffset(), curTopInterval.upperOffset());
        Interval splitHighInterval(curTopInterval.upperLevel(), curLowInterval.upperLevel(),
                                   curTopInterval.upperOffset() + 1, curLowInterval.upperOffset());

        *curLowIt = splitLowInterval;
        *curTopIt = splitMidInterval;
        newIntervals.insert(curTopIt, splitHighInterval);

        change = true;
      }
      // otherwise we will generate three intervals: the intersection and the two disjoint intervals
      // of the XOR
      // [a---------b]
      //   [c---------------d]
      // ====>>
      // [a--c-1]
      //        [c--b]
      //             [b+1--d]
      else if(curLowInterval.overlaps(curTopInterval)) {
        Interval splitLowInterval(curLowInterval.lowerLevel(), curTopInterval.lowerLevel(),
                                  curLowInterval.lowerOffset(), curTopInterval.lowerOffset() - 1);
        Interval splitMidInterval(curTopInterval.lowerLevel(), curLowInterval.upperLevel(),
                                  curTopInterval.lowerOffset(), curLowInterval.upperOffset());
        Interval splitHighInterval(curLowInterval.upperLevel(), curTopInterval.upperLevel(),
                                   curLowInterval.upperOffset() + 1, curTopInterval.upperOffset());

        *curLowIt = splitLowInterval;
        *curTopIt = splitMidInterval;
        newIntervals.insert(curTopIt, splitHighInterval);
        change = true;
      }

      // if we created a new interval, the order order of the list can be false now: assume
      // a : [0,2], b : [0,3], c : [1,5]
      // after the first iteration, we have:
      // a' : [0,2], b' : [3,3], c : [1,5]
      // and now b' > c. In this case the algorithm breaks.
      // In order to solve this, we sort the complete list once more and start from the start. This
      // is somewhat expensive but should not be too bad since the further we go, nothing happens to
      // the start of the array anymore.
      if(change) {
        std::sort(
            newIntervals.begin(), newIntervals.end(),
            [](Interval const& a, Interval const& b) { return a.lowerBound() < b.lowerBound(); });
        curLowIt = newIntervals.begin();
        curTopIt = std::next(curLowIt);
        cnt = 0;
        change = false;
      }
      // if no intervals were added, hence the only action was removing redundant intervals or
      // having non overlapping (and non containing) intervals, we just move up the array and find
      // the next pair of intervals to compare
      else {
        curLowIt = curTopIt;
        curTopIt++;
        cnt++;
      }
    }
  }
  return newIntervals;
}

} // namespace dawn
