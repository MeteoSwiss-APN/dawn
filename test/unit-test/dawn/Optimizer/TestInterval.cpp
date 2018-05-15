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

#include "dawn/Optimizer/IntervalAlgorithms.h"
#include "dawn/Optimizer/MultiInterval.h"
#include "dawn/Optimizer/Interval.h"
#include <gtest/gtest.h>
#include <unordered_set>

using namespace dawn;

namespace {

TEST(IntervalTest, Comparison) {
  Interval I0(0, 5);
  Interval I1(1, 5);

  EXPECT_TRUE(I0 == I0);
  EXPECT_TRUE(I0 != I1);
}

TEST(IntervalTest, Contains) {
  Interval I0(0, 5);
  Interval I1(1, 5);
  Interval I2(0, 4);
  Interval I3(1, 4);

  EXPECT_TRUE(I0.contains(I1));
  EXPECT_TRUE(I0.contains(I2));
  EXPECT_TRUE(I0.contains(I3));

  EXPECT_FALSE(I1.contains(I0));
  EXPECT_FALSE(I2.contains(I0));
  EXPECT_FALSE(I3.contains(I0));
}

TEST(IntervalTest, Overlaps) {
  Interval I0(0, 5);
  Interval I1(1, 5);
  Interval I2(0, 2);
  Interval I3(2, 5);
  Interval I4(3, 5);

  EXPECT_TRUE(I0.overlaps(I1));
  EXPECT_TRUE(I0.overlaps(I2));
  EXPECT_TRUE(I0.overlaps(I3));
  EXPECT_TRUE(I0.overlaps(I4));

  EXPECT_TRUE(I2.overlaps(I3));
  EXPECT_FALSE(I2.overlaps(I4));
}

TEST(IntervalTest, Adjacent) {
  Interval I0(0, 5);
  Interval I1(1, 5);
  Interval I2(0, 0);
  Interval I3(5, 5);
  Interval I4(6, 10);

  EXPECT_TRUE(I0.adjacent(I4));
  EXPECT_TRUE(I1.adjacent(I2));
  EXPECT_TRUE(I3.adjacent(I4));
  EXPECT_FALSE(I0.adjacent(I0));
  EXPECT_FALSE(I0.adjacent(I3));
}

TEST(IntervalTest, Merge1) {
  Interval I(0, 5, 0, 0);
  Interval IntervalToMerge(1, 5, -1, +1);

  I.merge(IntervalToMerge);
  EXPECT_TRUE(I == Interval(0, 5, 0, 1));
}

TEST(IntervalTest, Merge2) {
  Interval I(0, 80, -1, +1);
  Interval IntervalToMerge(0, 1, 0, 0);

  I.merge(IntervalToMerge);
  EXPECT_TRUE(I == Interval(0, 80, -1, +1));
}

TEST(IntervalTest, Merge3) {
  Interval I(0, 80, 0, -1);
  Interval IntervalToMerge(0, 79, 0, +2);

  I.merge(IntervalToMerge);
  EXPECT_TRUE(I == Interval(0, 80, 0, +1));
}

TEST(IntervalTest, LevelUnion1) {
  Interval I1(0, 0);

  auto levelUnion = Interval::computeLevelUnion(std::vector<Interval>{I1});
  EXPECT_TRUE((std::unordered_set<Interval>(levelUnion.begin(), levelUnion.end()) ==
               std::unordered_set<Interval>{Interval(0, 0)}));
}

TEST(IntervalTest, LevelUnion2) {
  Interval I1(0, 2);

  auto levelUnion = Interval::computeLevelUnion(std::vector<Interval>{I1});
  EXPECT_TRUE((std::unordered_set<Interval>(levelUnion.begin(), levelUnion.end()) ==
               std::unordered_set<Interval>{Interval(0, 0), Interval(1, 1), Interval(2, 2)}));
}

TEST(IntervalTest, LevelUnion3) {
  Interval I1(0, 3);

  auto levelUnion = Interval::computeLevelUnion(std::vector<Interval>{I1});
  EXPECT_TRUE((std::unordered_set<Interval>(levelUnion.begin(), levelUnion.end()) ==
               std::unordered_set<Interval>{Interval(0, 0), Interval(1, 2), Interval(3, 3)}));
}

TEST(IntervalTest, LevelUnion4) {
  Interval I1(0, 3);
  Interval I2(4, 5);

  auto levelUnion = Interval::computeLevelUnion(std::vector<Interval>{I1, I2});
  EXPECT_TRUE((std::unordered_set<Interval>(levelUnion.begin(), levelUnion.end()) ==
               std::unordered_set<Interval>{Interval(0, 0), Interval(1, 2), Interval(3, 3),
                                            Interval(4, 4), Interval(5, 5)}));
}

TEST(IntervalTest, LevelUnion5) {
  Interval I1(0, 20);
  Interval I2(5, 21);
  Interval I3(5, 10);
  Interval I4(21, 21);

  auto levelUnion = Interval::computeLevelUnion(std::vector<Interval>{I1, I2, I3, I4});
  EXPECT_TRUE((std::unordered_set<Interval>(levelUnion.begin(), levelUnion.end()) ==
               std::unordered_set<Interval>{
                   Interval(0, 0), Interval(1, 4), Interval(5, 5), Interval(6, 9), Interval(10, 10),
                   Interval(11, 19), Interval(20, 20), Interval(21, 21),
               }));
}

TEST(IntervalTest, GapIntervals1) {
  Interval Axis(0, 0);
  Interval I1(0, 0);

  auto gapIntervals = Interval::computeGapIntervals(Axis, std::vector<Interval>{I1});
  EXPECT_TRUE((std::unordered_set<Interval>(gapIntervals.begin(), gapIntervals.end()) ==
               std::unordered_set<Interval>{Interval(0, 0)} // [0, 0]
               ));
}

TEST(IntervalTest, GapIntervals2) {
  Interval Axis(0, 2);
  Interval I1(1, 2);

  auto gapIntervals = Interval::computeGapIntervals(Axis, std::vector<Interval>{I1});
  EXPECT_TRUE((std::unordered_set<Interval>(gapIntervals.begin(), gapIntervals.end()) ==
               std::unordered_set<Interval>{Interval(0, 1, 0, -1), // [0, 0]
                                            Interval(1, 2, 0, 0)}  // [1, 2]
               ));
}

TEST(IntervalTest, GapIntervals3) {
  Interval Axis(0, 2);
  Interval I1(0, 1);

  auto gapIntervals = Interval::computeGapIntervals(Axis, std::vector<Interval>{I1});
  EXPECT_TRUE((std::unordered_set<Interval>(gapIntervals.begin(), gapIntervals.end()) ==
               std::unordered_set<Interval>{Interval(0, 1, 0, 0), // [0, 1]
                                            Interval(1, 2, 1, 0)} // [2, 2]
               ));
}

TEST(IntervalTest, GapIntervals4) {
  Interval Axis(0, 6);
  Interval I1(1, 2);
  Interval I2(4, 5);

  auto gapIntervals = Interval::computeGapIntervals(Axis, std::vector<Interval>{I1, I2});
  EXPECT_TRUE((std::unordered_set<Interval>(gapIntervals.begin(), gapIntervals.end()) ==
               std::unordered_set<Interval>{Interval(0, 1, 0, -1), // [0, 0]
                                            Interval(1, 2, 0, 0),  // [1, 2]
                                            Interval(2, 4, 1, -1), // [3, 3]
                                            Interval(4, 5, 0, 0),  // [4, 5]
                                            Interval(5, 6, 1, 0)}  // [6, 6]
               ));
}

TEST(IntervalTest, GapIntervals5) {
  Interval Axis(0, 6);
  Interval I1(1, 2);
  Interval I2(4, 5);

  auto gapIntervals = Interval::computeGapIntervals(Axis, std::vector<Interval>{I1, I2});

  EXPECT_TRUE((std::unordered_set<Interval>(gapIntervals.begin(), gapIntervals.end()) ==
               std::unordered_set<Interval>{Interval(0, 1, 0, -1), // [0, 0]
                                            Interval(1, 2, 0, 0),  // [1, 2]
                                            Interval(2, 4, 1, -1), // [3, 3]
                                            Interval(4, 5, 0, 0),  // [4, 5]
                                            Interval(5, 6, 1, 0)}  // [6, 6]
               ));
}

TEST(IntervalTest, GapIntervals6) {
  Interval Axis(0, 20);
  Interval I1(1, 2);
  Interval I2(4, 5);
  Interval I3(10, 15);

  auto gapIntervals = Interval::computeGapIntervals(Axis, std::vector<Interval>{I1, I2, I3});

  EXPECT_TRUE((std::unordered_set<Interval>(gapIntervals.begin(), gapIntervals.end()) ==
               std::unordered_set<Interval>{Interval(0, 1, 0, -1),  // [0, 0]
                                            Interval(1, 2, 0, 0),   // [1, 2]
                                            Interval(2, 4, 1, -1),  // [3, 3]
                                            Interval(4, 5, 0, 0),   // [4, 5]
                                            Interval(5, 10, 1, -1), // [6, 9]
                                            Interval(10, 15, 0, 0), // [10, 15]
                                            Interval(15, 20, 1, 0)} // [16, 20]
               ));
}

TEST(IntervalTest, GapIntervals7) {
  Interval Axis(0, 20);
  Interval I1(1, 2);
  Interval I2(4, 5);
  Interval I3(10, 15);

  // Intervals given unordered!
  auto gapIntervals = Interval::computeGapIntervals(Axis, std::vector<Interval>{I3, I1, I2});

  EXPECT_TRUE((std::unordered_set<Interval>(gapIntervals.begin(), gapIntervals.end()) ==
               std::unordered_set<Interval>{Interval(0, 1, 0, -1),  // [0, 0]
                                            Interval(1, 2, 0, 0),   // [1, 2]
                                            Interval(2, 4, 1, -1),  // [3, 3]
                                            Interval(4, 5, 0, 0),   // [4, 5]
                                            Interval(5, 10, 1, -1), // [6, 9]
                                            Interval(10, 15, 0, 0), // [10, 15]
                                            Interval(15, 20, 1, 0)} // [16, 20]
               ));
}

TEST(IntervalTest, GapIntervalsOverlap1) {
  Interval Axis(0, 6);
  Interval I1(1, 4);
  Interval I2(2, 5);

  EXPECT_DEATH(Interval::computeGapIntervals(Axis, std::vector<Interval>{I1, I2}), ".*");
}

TEST(IntervalTest, GapIntervalsOverlap2) {
  Interval Axis(0, 6);
  Interval I1(1, 8);
  Interval I2(5, 10);

  EXPECT_DEATH(Interval::computeGapIntervals(Axis, std::vector<Interval>{I1, I2}), ".*");
}

TEST(IntervalTest, GapIntervalsOverlap3) {
  Interval Axis(0, 6);
  Interval I1(6, 7);
  Interval I2(5, 6);

  EXPECT_DEATH(Interval::computeGapIntervals(Axis, std::vector<Interval>{I1, I2}), ".*");
}

TEST(IntervalTest, PartitionIntervals0) {
  Interval I1(1, 5);
  Interval I2(1, 3);

  auto partIntervals = Interval::computePartition(std::vector<Interval>{I1, I2});
  std::unordered_set<Interval> solution(partIntervals.begin(), partIntervals.end());
  std::unordered_set<Interval> reference{Interval(1, 3), Interval(4, 5)};

  EXPECT_TRUE((reference == solution));
}

TEST(IntervalTest, PartitionIntervals1) {
  Interval I1(1, 5);
  Interval I2(1, 5);

  auto partIntervals = Interval::computePartition(std::vector<Interval>{I1, I2});
  std::unordered_set<Interval> solution(partIntervals.begin(), partIntervals.end());
  std::unordered_set<Interval> reference{Interval(1, 5)};

  EXPECT_TRUE((reference == solution));
}

TEST(IntervalTest, PartitionIntervals2) {
  Interval I1(1, 3);
  Interval I2(2, 5);

  auto partIntervals = Interval::computePartition(std::vector<Interval>{I1, I2});
  std::unordered_set<Interval> solution(partIntervals.begin(), partIntervals.end());
  std::unordered_set<Interval> reference{Interval(1, 1), Interval(2, 3), Interval(4, 5)};

  EXPECT_TRUE((reference == solution));
}

TEST(IntervalTest, PartitionIntervals3) {
  Interval I1(1, 5);
  Interval I2(2, 9);

  auto partIntervals = Interval::computePartition(std::vector<Interval>{I1, I2});
  std::unordered_set<Interval> solution(partIntervals.begin(), partIntervals.end());
  std::unordered_set<Interval> reference{Interval(1, 1), Interval(2, 5), Interval(6, 9)};

  EXPECT_TRUE((reference == solution));
}

TEST(IntervalTest, PartitionIntervals4) {
  Interval I1(1, 5);
  Interval I2(2, 9);
  Interval I3(4, 7);

  auto partIntervals = Interval::computePartition(std::vector<Interval>{I1, I2, I3});
  std::unordered_set<Interval> solution(partIntervals.begin(), partIntervals.end());
  std::unordered_set<Interval> reference{Interval(1, 1), Interval(2, 3), Interval(4, 5),
                                         Interval(6, 7), Interval(8, 9)};

  EXPECT_TRUE((reference == solution));
}

TEST(IntervalTest, Construction) {
  Interval I0(sir::Interval::Start, sir::Interval::End);
  Interval I1(sir::Interval::Start, sir::Interval::End, -1, -2);
  Interval I2(0, sir::Interval::End, -4, +2);

  EXPECT_EQ(I0.lowerBound(), sir::Interval::Start);
  EXPECT_EQ(I0.upperBound(), sir::Interval::End);
  EXPECT_EQ(I1.lowerBound(), sir::Interval::Start - 1);
  EXPECT_EQ(I1.upperBound(), sir::Interval::End - 2);
  EXPECT_EQ(I2.lowerBound(), -4);
  EXPECT_EQ(I2.upperBound(), sir::Interval::End + 2);

  // Copy assign
  I0 = I2;
  EXPECT_TRUE(I0 == I2);

  // Move assign
  I0 = std::move(I1);
  EXPECT_TRUE(I0 == I1);
}

TEST(IntervalTest, Substract) {
  {
    // I1 & I2 do not overlap
    Interval I1(6, 7);
    Interval I2(4, 5);

    EXPECT_EQ((substract(I1, I2)), MultiInterval{I1});
  }
  {
    // overlap in one level
    Interval I1(6, 7);
    Interval I2(5, 6);

    EXPECT_EQ((substract(I1, I2)), (MultiInterval{Interval{6, 7, 1, 0}}));
  }
  {
    Interval I1(6, 10);
    Interval I2(5, 6, 0, 2);

    EXPECT_EQ((substract(I1, I2)), (MultiInterval{Interval{6, 10, 3, 0}}));
  }
  {
    // I2 contains I1
    Interval I1(6, 7);
    Interval I2(5, 6, 0, 2);

    EXPECT_TRUE(substract(I1, I2).getIntervals().size() == 0);
  }
  {
    Interval I1(4, 7);
    Interval I2(5, 8);

    // TODO what if I2 is inside I1? We need a multi interval
    EXPECT_EQ((substract(I1, I2)), (MultiInterval{Interval{4, 4}}));
  }
  {
    Interval I1{0, sir::Interval::End - 4};
    Interval I2{1, sir::Interval::End - 4};

    EXPECT_EQ((substract(I1, I2)), (MultiInterval{Interval{0, 0}}));
  }
}

} // anonymous namespace
