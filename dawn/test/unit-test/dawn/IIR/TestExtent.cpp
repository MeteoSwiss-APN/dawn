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

#include "dawn/IIR/Extents.h"
#include <gtest/gtest.h>

using namespace dawn;
using namespace iir;
namespace {

TEST(ExtentsTest, Construction) {
  Extents extents(ast::Offsets{ast::cartesian, -1, 1, 0});

  EXPECT_EQ(extents[0], (Extent{-1, -1}));
  EXPECT_EQ(extents[1], (Extent{1, 1}));
  EXPECT_EQ(extents[2], (Extent{0, 0}));
}

TEST(ExtentsTest, PointWise) {
  Extents extents1(ast::Offsets{ast::cartesian, 0, 1, 0});
  EXPECT_FALSE(extents1.isHorizontalPointwise());
  EXPECT_TRUE(extents1.isVerticalPointwise());

  Extents extents2(ast::Offsets{ast::cartesian, 0, 0, 1});
  EXPECT_TRUE(extents2.isHorizontalPointwise());
  EXPECT_FALSE(extents2.isVerticalPointwise());
}

TEST(ExtentsTest, Merge1) {
  Extents extents(ast::Offsets{ast::cartesian, -1, 1, 0});
  Extents extentsToMerge(ast::Offsets{ast::cartesian, 3, 2, 1});
  extents.merge(extentsToMerge);

  EXPECT_TRUE((extents[0] == Extent{-1, 3}));
  EXPECT_TRUE((extents[1] == Extent{1, 2}));
  EXPECT_TRUE((extents[2] == Extent{0, 1}));
}

TEST(ExtentsTest, Merge2) {
  Extents extents(ast::Offsets{ast::cartesian, -1, 1, 0});
  Extents extentsToMerge(ast::Offsets{ast::cartesian, -2, 2, 0});
  extents.merge(extentsToMerge);

  EXPECT_TRUE((extents[0] == Extent{-2, -1}));
  EXPECT_TRUE((extents[1] == Extent{1, 2}));
  EXPECT_TRUE((extents[2] == Extent{0, 0}));
}

TEST(ExtentsTest, Merge3) {
  Extents extents(ast::Offsets{ast::cartesian, -1, 1, 0});
  extents.merge(ast::Offsets{ast::cartesian, -2, 0, 0});

  EXPECT_TRUE((extents[0] == Extent{-2, -1}));
  EXPECT_TRUE((extents[1] == Extent{0, 1}));
  EXPECT_TRUE((extents[2] == Extent{0, 0}));
}

TEST(ExtentsTest, Add1) {
  Extents extents(-1, 1, 0, 0, 0, 0);
  extents.add(ast::Offsets{ast::cartesian, 1, 0, 0});

  EXPECT_TRUE((extents[0] == Extent{0, 2}));
  EXPECT_TRUE((extents[1] == Extent{0, 0}));
  EXPECT_TRUE((extents[2] == Extent{0, 0}));
}

TEST(ExtentsTest, Add2) {
  Extents extents(0, 1, 0, 0, 0, 0);
  extents.add(ast::Offsets{ast::cartesian, -1, 0, 0});

  EXPECT_TRUE((extents[0] == Extent{-1, 0}));
  EXPECT_TRUE((extents[1] == Extent{0, 0}));
  EXPECT_TRUE((extents[2] == Extent{0, 0}));
}

TEST(ExtentsTest, Add3) {
  Extents extents(-2, 2, 0, 0, 0, 0);
  extents.add(ast::Offsets{ast::cartesian, 1, 0, 0});

  EXPECT_TRUE((extents[0] == Extent{-1, 3}));
  EXPECT_TRUE((extents[1] == Extent{0, 0}));
  EXPECT_TRUE((extents[2] == Extent{0, 0}));
}

TEST(ExtentsTest, Add4) {
  Extents extents(-2, 2, 0, 0, 0, 0);
  extents.add({-2, 2, 0, 0, 0, 0});

  EXPECT_TRUE((extents[0] == Extent{-4, 4}));
  EXPECT_TRUE((extents[1] == Extent{0, 0}));
  EXPECT_TRUE((extents[2] == Extent{0, 0}));
}

TEST(ExtentsTest, Add5) {
  Extents extents({-2, 2, 0, 4, 0, 0});
  extents.add(ast::Offsets{ast::cartesian, 2, 2, 3});

  EXPECT_TRUE((extents[0] == Extent{0, 4}));
  EXPECT_TRUE((extents[1] == Extent{0, 6}));
  EXPECT_TRUE((extents[2] == Extent{0, 3}));
}

TEST(ExtentsTest, addCenter) {
  Extents extents({1, 1, -2, -2, 3, 3});
  extents.addCenter(0);

  EXPECT_EQ(extents, (Extents{0, 1, -2, -2, 3, 3}));
  extents.addCenter(1);
  EXPECT_EQ(extents, (Extents{0, 1, -2, 0, 3, 3}));
}

TEST(ExtentsTest, Stringify) {
  Extents extents(ast::Offsets{ast::cartesian, 1, -1, 2});
  std::stringstream ss;
  ss << extents;
  EXPECT_STREQ(ss.str().c_str(), "[(1, 1), (-1, -1), (2, 2)]");
}

TEST(ExtentsTest, verticalLoopOrder) {
  Extents extents{0, 0, 0, 0, -1, 2};
  EXPECT_TRUE((extents.getVerticalLoopOrderExtent(
                  iir::LoopOrderKind::Forward,
                  Extents::VerticalLoopOrderDir::CounterLoopOrder, false)) == (Extent{1, 2}));
  EXPECT_TRUE((extents.getVerticalLoopOrderExtent(
                  iir::LoopOrderKind::Forward,
                  Extents::VerticalLoopOrderDir::CounterLoopOrder, true)) == (Extent{0, 2}));
  EXPECT_TRUE((extents.getVerticalLoopOrderExtent(iir::LoopOrderKind::Forward,
                                                  Extents::VerticalLoopOrderDir::InLoopOrder,
                                                  false)) == (Extent{-1, -1}));
  EXPECT_TRUE((extents.getVerticalLoopOrderExtent(iir::LoopOrderKind::Forward,
                                                  Extents::VerticalLoopOrderDir::InLoopOrder,
                                                  true)) == (Extent{-1, 0}));

  EXPECT_TRUE((extents.getVerticalLoopOrderExtent(
                  iir::LoopOrderKind::Backward,
                  Extents::VerticalLoopOrderDir::CounterLoopOrder, false)) == (Extent{-1, -1}));
  EXPECT_TRUE((extents.getVerticalLoopOrderExtent(
                  iir::LoopOrderKind::Backward,
                  Extents::VerticalLoopOrderDir::CounterLoopOrder, true)) == (Extent{-1, 0}));
  EXPECT_TRUE((extents.getVerticalLoopOrderExtent(iir::LoopOrderKind::Backward,
                                                  Extents::VerticalLoopOrderDir::InLoopOrder,
                                                  false)) == (Extent{1, 2}));
  EXPECT_TRUE((extents.getVerticalLoopOrderExtent(iir::LoopOrderKind::Backward,
                                                  Extents::VerticalLoopOrderDir::InLoopOrder,
                                                  true)) == (Extent{0, 2}));
}
} // anonymous namespace
