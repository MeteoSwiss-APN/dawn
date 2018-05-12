#include "dawn/Optimizer/IntervalAlgorithms.h"

namespace dawn {
MultiInterval substract(const Interval& int1, const Interval& int2) {
  std::cout << "OP " << int1 << " " << int2 << std::endl;

  if(!int1.overlaps(int2))
    return MultiInterval({int1});
  std::cout << "INTO " << std::endl;
  if(int2.contains(int1))
    return MultiInterval();
  std::cout << "check " << std::endl;
  if(int1.contains(int2) && (int1.lowerBound() != int2.lowerBound()) &&
     (int1.upperBound() != int2.upperBound())) {
    std::cout << "OP K" << int1 << " " << int2 << std::endl;
    auto ii = MultiInterval{
        Interval{int1.lowerLevel(), int2.lowerLevel(), int1.lowerOffset(), int2.lowerOffset() - 1},
        Interval{int2.upperLevel(), int1.upperLevel(), int2.upperOffset() + 1, int1.upperOffset()}};

    std::cout << "IO " << std::endl;
    std::cout << "OOP " << ii << std::endl;

    return MultiInterval{
        Interval{int1.lowerLevel(), int2.lowerLevel(), int1.lowerOffset(), int2.lowerOffset() - 1},
        Interval{int2.upperLevel(), int1.upperLevel(), int2.upperOffset() + 1, int1.upperOffset()}};
  }

  bool upperI2InInterval =
      (int2.upperBound() >= int1.lowerBound()) && (int2.upperBound() < int1.upperBound());
  bool lowerI2InInterval =
      (int2.lowerBound() > int1.lowerBound()) && (int2.lowerBound() <= int1.upperBound());

  std::cout << " R " << std::endl;
  return MultiInterval{Interval{(upperI2InInterval ? int2.upperLevel() : int1.lowerLevel()),
                                (lowerI2InInterval ? int2.lowerLevel() : int1.upperLevel()),
                                (upperI2InInterval ? int2.upperOffset() + 1 : int1.lowerOffset()),
                                (lowerI2InInterval ? int2.lowerOffset() - 1 : int1.upperOffset())}};
}

// TODO unittest this
MultiInterval substract(const Interval& int1, const MultiInterval& int2) {
  if(int2.empty())
    return MultiInterval{int1};

  MultiInterval result = substract(int1, int2.getIntervals().front());

  if(int2.numPartitions() == 1)
    return result;

  for(auto it = ++(int2.getIntervals().begin()); it != int2.getIntervals().end(); ++it) {
    result.substract(*it);
  }
  return result;
}

// TODO Unittest this
Cache::window computeWindowOffset(LoopOrderKind loopOrder, Interval const& accessInterval,
                                  Interval const& computeInterval) {
  std::cout << "For " << loopOrder << " " << accessInterval << " " << computeInterval << " "
            << accessInterval.lowerBound() << " " << computeInterval.upperBound() << std::endl;
  return Cache::window{(accessInterval.lowerBound() - ((loopOrder == LoopOrderKind::LK_Backward)
                                                           ? computeInterval.upperBound()
                                                           : computeInterval.lowerBound())),
                       (accessInterval.upperBound() - ((loopOrder == LoopOrderKind::LK_Backward)
                                                           ? computeInterval.upperBound()
                                                           : computeInterval.lowerBound()))};
}
}
