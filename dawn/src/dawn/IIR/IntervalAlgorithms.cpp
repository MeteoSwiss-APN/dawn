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

#include "dawn/IIR/IntervalAlgorithms.h"

namespace dawn {
namespace iir {
MultiInterval substract(const iir::Interval& int1, const iir::Interval& int2) {

  if(!int1.overlaps(int2))
    return MultiInterval({int1});
  if(int2.contains(int1))
    return MultiInterval();
  if(int1.contains(int2) && (int1.lowerBound() != int2.lowerBound()) &&
     (int1.upperBound() != int2.upperBound())) {

    return MultiInterval{iir::Interval{int1.lowerLevel(), int2.lowerLevel(), int1.lowerOffset(),
                                       int2.lowerOffset() - 1},
                         iir::Interval{int2.upperLevel(), int1.upperLevel(), int2.upperOffset() + 1,
                                       int1.upperOffset()}};
  }

  bool upperI2InInterval =
      (int2.upperBound() >= int1.lowerBound()) && (int2.upperBound() < int1.upperBound());
  bool lowerI2InInterval =
      (int2.lowerBound() > int1.lowerBound()) && (int2.lowerBound() <= int1.upperBound());

  return MultiInterval{
      iir::Interval{(upperI2InInterval ? int2.upperLevel() : int1.lowerLevel()),
                    (lowerI2InInterval ? int2.lowerLevel() : int1.upperLevel()),
                    (upperI2InInterval ? int2.upperOffset() + 1 : int1.lowerOffset()),
                    (lowerI2InInterval ? int2.lowerOffset() - 1 : int1.upperOffset())}};
}

MultiInterval substract(const iir::Interval& int1, const MultiInterval& int2) {
  if(int2.empty())
    return MultiInterval{int1};

  MultiInterval result = MultiInterval{int1};

  for(auto it = int2.getIntervals().begin(); it != int2.getIntervals().end(); ++it) {
    result.substract(*it);
  }
  return result;
}

Cache::window computeWindowOffset(LoopOrderKind loopOrder, iir::Interval const& accessInterval,
                                  iir::Interval const& computeInterval) {
  return Cache::window{(accessInterval.lowerBound() - ((loopOrder == LoopOrderKind::LK_Backward)
                                                           ? computeInterval.upperBound()
                                                           : computeInterval.lowerBound())),
                       (accessInterval.upperBound() - ((loopOrder == LoopOrderKind::LK_Backward)
                                                           ? computeInterval.upperBound()
                                                           : computeInterval.lowerBound()))};
}
} // namespace iir
} // namespace dawn
