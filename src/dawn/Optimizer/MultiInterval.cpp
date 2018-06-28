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

#include "dawn/Optimizer/MultiInterval.h"
#include "dawn/Optimizer/IntervalAlgorithms.h"

namespace dawn {

std::ostream& operator<<(std::ostream& os, const MultiInterval& multiInterval) {
  for(auto const& interv : multiInterval.getIntervals()) {
    os << "[ " << interv << " ]";
  }

  return os;
}

MultiInterval::MultiInterval(std::initializer_list<Interval> const& intervals)
    : intervals_(intervals) {}

void MultiInterval::insert(MultiInterval const& multiInterval) {
  for(auto const& interv : multiInterval.getIntervals())
    insert(interv);
}

bool MultiInterval::operator==(const MultiInterval& other) const {
  if(other.getIntervals().size() != intervals_.size())
    return false;
  for(int i = 0; i < intervals_.size(); ++i) {
    if(intervals_[i] != other.getIntervals()[i])
      return false;
  }
  return true;
}

void MultiInterval::insert(boost::optional<Interval> const& interval) {
  if(interval.is_initialized())
    insert(*interval);
}

void MultiInterval::substract(Interval const& interval) {
  for(auto it = intervals_.begin(); it != intervals_.end(); ++it) {
    if(it->overlaps(interval)) {
      const Interval int1 = *it;
      MultiInterval multiInterval = dawn::substract(int1, interval);

      intervals_.erase(it);

      for(auto const& intervIt : multiInterval.getIntervals()) {
        insert(intervIt);
      }
      it = intervals_.begin();
      if(it == intervals_.end())
        break;
    }
  }
}

void MultiInterval::substract(MultiInterval const& multiInterval) {
  for(auto const& interv : multiInterval.getIntervals()) {
    substract(interv);
  }
}

void MultiInterval::insert(Interval const& interval) {
  intervals_.push_back(interval);
  auto intervals = Interval::computePartition(intervals_);
  intervals_.clear();
  std::move(intervals.begin(), intervals.end(), std::back_inserter(intervals_));

  for(auto it = intervals_.begin(); it != intervals_.end(); ++it) {
    if(std::next(it) == intervals_.end())
      break;
    auto& aInterval = *it;
    auto& nextInterval = *(std::next(it));
    if(aInterval.adjacent(nextInterval)) {
      aInterval.merge(nextInterval);

      it = std::prev(std::prev(intervals_.erase(std::next(it))));
    }
  }
}
} // namespace dawn
