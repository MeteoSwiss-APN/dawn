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

TEST(IntervalAlgorithms, substractIntervals) {
  {
    Interval I0(0, 5);
    Interval I1(1, 5);

    auto res = substract(I0, I1);
    EXPECT_EQ(res, (MultiInterval{Interval{0, 0}}));
  }

  {
    Interval I0(0, 5);
    Interval I1(0, 5);

    auto res = substract(I0, I1);
    EXPECT_EQ(res, (MultiInterval{}));
  }

  {
    Interval I0(0, 5);
    Interval I1(6, 7);

    auto res = substract(I0, I1);
    EXPECT_EQ(res, (MultiInterval{I0}));
  }

  {
    Interval I0(4, 7);
    Interval I1(2, 5);

    auto res = substract(I0, I1);
    EXPECT_EQ(res, (MultiInterval{Interval{6, 7}}));
  }
}

TEST(IntervalAlgorithms, substractIntervalsMultiInterval) {
  {
    Interval I0{0, 9};
    MultiInterval I1{Interval{2, 4}, Interval{6, 8}};

    auto res = substract(I0, I1);
    EXPECT_EQ(res, (MultiInterval{Interval{0, 1}, Interval{5, 5}, Interval{9, 9}}));
  }

  {
    Interval I0{0, 9};
    MultiInterval I1{Interval{2, 4}, Interval{6, 10}};

    auto res = substract(I0, I1);
    EXPECT_EQ(res, (MultiInterval{Interval{0, 1}, Interval{5, 5}}));
  }
}

TEST(IntervalAlgorithms, computeWindowOffset) {

  EXPECT_EQ((Cache::window{-3, 0}),
            (computeWindowOffset(LoopOrderKind::LK_Forward, Interval{-2, 1}, Interval{1, 10})));
  EXPECT_EQ((Cache::window{0, 2}), (computeWindowOffset(LoopOrderKind::LK_Backward,
                                                        Interval{10, 10 + 2}, Interval{0, 10})));
}

} // anonymous namespace
