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

#include "dawn/IIR/Interval.h"
#include <list>
#include <optional>

namespace dawn {
namespace iir {

class MultiInterval {
  std::vector<iir::Interval> intervals_;

public:
  /// @name Constructors and Assignment
  MultiInterval() = default;
  MultiInterval(const std::vector<Interval>& intervals);
  MultiInterval(std::initializer_list<Interval> const& intervals);

  void insert(iir::Interval const& interval);
  void insert(std::optional<iir::Interval> const& interval);
  void insert(MultiInterval const& multiInterval);
  void substract(iir::Interval const& interval);
  void substract(MultiInterval const& multiInterval);
  bool overlaps(const Interval& other) const;
  bool contiguous() const;

  std::vector<iir::Interval> const& getIntervals() const { return intervals_; }

  bool empty() const { return (intervals_.size() == 0); }
  bool operator==(const MultiInterval& other) const;

  bool operator!=(const MultiInterval& other) const { return !(*this == other); }

  int numPartitions() const { return intervals_.size(); }

  friend std::ostream& operator<<(std::ostream& os, const MultiInterval& interval);
};

} // namespace iir
} // namespace dawn
