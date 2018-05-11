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

#ifndef DAWN_OPTIMIZER_MULTIINTERVAL_H
#define DAWN_OPTIMIZER_MULTIINTERVAL_H

#include "dawn/Optimizer/Interval.h"
#include <list>

namespace dawn {

class MultiInterval {
  std::vector<Interval> intervals_;

public:
  /// @name Constructors and Assignment
  MultiInterval() = default;
  MultiInterval(std::initializer_list<Interval> const& intervals);

  void insert(Interval const& interval);
  void insert(boost::optional<Interval> const& interval);
  void insert(MultiInterval const& multiInterval);
  void substract(Interval const& interval);

  std::vector<Interval> const& getIntervals() const { return intervals_; }

  bool empty() const { return (intervals_.size() == 0); }
  bool operator==(const MultiInterval& other) const;

  bool operator!=(const MultiInterval& other) const { return !(*this == other); }

  int numPartitions() const { return intervals_.size(); }

  friend std::ostream& operator<<(std::ostream& os, const MultiInterval& interval);
};

} // namespace dawn

#endif
