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
#include <gtest/gtest.h>

using namespace dawn;

namespace {

TEST(MultIntervalTest, Insert) {
  {
    MultiInterval multiInterval;

    multiInterval.insert({0, 5});
    multiInterval.insert({6, 9});

    EXPECT_TRUE(multiInterval.getIntervals().size() == 1);
    EXPECT_EQ(multiInterval.getIntervals()[0], (Interval{0, 9}));
  }
  {
    MultiInterval multiInterval;

    multiInterval.insert({0, 5});
    multiInterval.insert({5, 9});

    EXPECT_TRUE(multiInterval.getIntervals().size() == 1);
    EXPECT_EQ(multiInterval.getIntervals()[0], (Interval{0, 9}));
  }
  {
    MultiInterval multiInterval;

    multiInterval.insert({0, 5});
    multiInterval.insert({7, 9});

    EXPECT_TRUE(multiInterval.getIntervals().size() == 2);
    EXPECT_EQ(multiInterval.getIntervals()[0], (Interval{0, 5}));
    EXPECT_EQ(multiInterval.getIntervals()[1], (Interval{7, 9}));
  }
  {
    MultiInterval multiInterval;

    multiInterval.insert({0, 5});
    multiInterval.insert({2, 3});

    EXPECT_TRUE(multiInterval.getIntervals().size() == 1);
    EXPECT_EQ(multiInterval.getIntervals()[0], (Interval{0, 5}));
  }
}

TEST(MultIntervalTest, Substact) {
  {
    MultiInterval multiInterval{Interval{0, 5}, Interval{7, 9}};

    multiInterval.substract(Interval{4, 7});
    EXPECT_TRUE(multiInterval.getIntervals().size() == 2);
    EXPECT_EQ(multiInterval.getIntervals()[0], (Interval{0, 3}));
    EXPECT_EQ(multiInterval.getIntervals()[1], (Interval{8, 9}));
  }
  {
    MultiInterval multiInterval{Interval{0, 5}, Interval{7, 9}};

    multiInterval.substract(Interval{0, 5});
    EXPECT_TRUE(multiInterval.getIntervals().size() == 1);
    EXPECT_EQ(multiInterval.getIntervals()[0], (Interval{7, 9}));
  }
  {
    MultiInterval multiInterval{Interval{0, 5}, Interval{4, 9}};

    multiInterval.substract(Interval{2, 3});
    EXPECT_TRUE(multiInterval.getIntervals().size() == 2);
    EXPECT_EQ(multiInterval.getIntervals()[0], (Interval{0, 1}));
    EXPECT_EQ(multiInterval.getIntervals()[1], (Interval{4, 9}));
  }
}
} // anonymous namespace
