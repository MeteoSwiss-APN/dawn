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
  Extents expected(ast::cartesian, -1, -1, 1, 1, 2, 2);

  Extents extents(ast::Offsets{ast::cartesian, -1, 1, 2});
  EXPECT_EQ(extents, expected);

  Extents extents2 = extents;
  EXPECT_EQ(extents2, expected);

  Extents extents3 = std::move(extents2);
  EXPECT_EQ(extents3, expected);

  Extents extents4{extents3};
  EXPECT_EQ(extents3, expected);
  EXPECT_EQ(extents4, expected);

  Extents extents5{std::move(extents4)};
  EXPECT_EQ(extents5, expected);
}
TEST(ExtentsTest, EmptyConstruction) {
  EXPECT_EQ(Extents(ast::cartesian), Extents());
  EXPECT_NE(Extents(ast::cartesian, -1, 1, -1, 1, -1, 1), Extents());

  EXPECT_EQ(Extents(), Extents());

  Extents nullExtent;
  Extents extents1{ast::cartesian, -1, 1, -1, 1, -1, 1};
  extents1 = nullExtent;
  EXPECT_EQ(extents1, nullExtent);

  Extents extents2{ast::cartesian, -1, 1, -1, 1, -1, 1};
  extents2 = std::move(nullExtent);
  EXPECT_EQ(extents2, nullExtent);
}

TEST(ExtentsTest, PointWise) {
  Extents extents1(ast::Offsets{ast::cartesian, 0, 1, 0});
  EXPECT_FALSE(extents1.isHorizontalPointwise());
  EXPECT_TRUE(extents1.isVerticalPointwise());

  Extents extents2(ast::Offsets{ast::cartesian, 1, 0, 0});
  EXPECT_FALSE(extents2.isHorizontalPointwise());
  EXPECT_TRUE(extents2.isVerticalPointwise());

  Extents extents3(ast::Offsets{ast::cartesian, 0, 0, 1});
  EXPECT_TRUE(extents3.isHorizontalPointwise());
  EXPECT_FALSE(extents3.isVerticalPointwise());

  EXPECT_TRUE(Extents().isPointwise());
}

TEST(ExtentsTest, MergeWithExtent) {
  Extents extents(ast::Offsets{ast::cartesian, -1, 1, 0});
  Extents extentsToMerge(ast::Offsets{ast::cartesian, 3, 2, 1});
  extents.merge(extentsToMerge);

  EXPECT_EQ(extents, Extents(ast::cartesian, -1, 3, 1, 2, 0, 1));

  Extents emptyExtent;
  emptyExtent.merge(Extents());
  EXPECT_EQ(emptyExtent, Extents());
  emptyExtent.merge(Extents(ast::cartesian, -1, -1, 1, 1, 0, 0));
  EXPECT_EQ(emptyExtent, Extents(ast::cartesian, -1, 0, 0, 1, 0, 0));
  emptyExtent.merge(Extents());
  EXPECT_EQ(emptyExtent, Extents(ast::cartesian, -1, 0, 0, 1, 0, 0));
}

TEST(ExtentsTest, MergeWithOffset) {
  Extents extents(ast::Offsets{ast::cartesian, -1, 1, 0});
  extents.merge(ast::Offsets{ast::cartesian, -2, 2, 0});
  EXPECT_EQ(extents, Extents(ast::cartesian, -2, -1, 1, 2, 0, 0));

  Extents emptyExtent;
  emptyExtent.merge(ast::Offsets{ast::cartesian, 0, 0, 0});
  EXPECT_EQ(emptyExtent, Extents());
  emptyExtent.merge(ast::Offsets{ast::cartesian, 1, 2, 3});
  EXPECT_EQ(emptyExtent, Extents(ast::cartesian, 0, 1, 0, 2, 0, 3));
}

TEST(ExtentsTest, Add) {
  Extents extents(dawn::ast::cartesian, -2, 2, 0, 0, 0, 0);
  EXPECT_EQ(extents + extents, Extents(ast::cartesian, -4, 4, 0, 0, 0, 0));
  extents += extents;
  EXPECT_EQ(extents, Extents(ast::cartesian, -4, 4, 0, 0, 0, 0));

  Extents emptyExtent;
  EXPECT_EQ(emptyExtent + extents, extents);
  EXPECT_EQ(extents + emptyExtent, extents);
  EXPECT_EQ(emptyExtent + emptyExtent, emptyExtent);
}

TEST(ExtentsTest, addCenter) {
  Extents extents(dawn::ast::cartesian, 1, 1, -2, -2, 3, 3);
  extents.addVerticalCenter();
  EXPECT_EQ(extents, Extents(ast::cartesian, 1, 1, -2, -2, 0, 3));
}

TEST(ExtentsTest, Stringify) {
  Extents extents(ast::Offsets{ast::cartesian, 1, -1, 2});
  std::stringstream ss;
  ss << extents;
  EXPECT_STREQ(ss.str().c_str(), "[(1, 1), (-1, -1), (2, 2)]");
}

TEST(ExtentsTest, verticalLoopOrder) {
  Extents extents(dawn::ast::cartesian, 0, 0, 0, 0, -1, 2);
  EXPECT_EQ(extents.getVerticalLoopOrderExtent(iir::LoopOrderKind::LK_Forward,
                                               Extents::VerticalLoopOrderDir::VL_CounterLoopOrder,
                                               false),
            Extent(1, 2));
  EXPECT_EQ(extents.getVerticalLoopOrderExtent(iir::LoopOrderKind::LK_Forward,
                                               Extents::VerticalLoopOrderDir::VL_CounterLoopOrder,
                                               true),
            Extent(0, 2));
  EXPECT_EQ(extents.getVerticalLoopOrderExtent(iir::LoopOrderKind::LK_Forward,
                                               Extents::VerticalLoopOrderDir::VL_InLoopOrder,
                                               false),
            Extent(-1, -1));
  EXPECT_EQ(extents.getVerticalLoopOrderExtent(iir::LoopOrderKind::LK_Forward,
                                               Extents::VerticalLoopOrderDir::VL_InLoopOrder, true),
            Extent(-1, 0));

  EXPECT_EQ(extents.getVerticalLoopOrderExtent(iir::LoopOrderKind::LK_Backward,
                                               Extents::VerticalLoopOrderDir::VL_CounterLoopOrder,
                                               false),
            Extent(-1, -1));
  EXPECT_EQ(extents.getVerticalLoopOrderExtent(iir::LoopOrderKind::LK_Backward,
                                               Extents::VerticalLoopOrderDir::VL_CounterLoopOrder,
                                               true),
            Extent(-1, 0));
  EXPECT_EQ(extents.getVerticalLoopOrderExtent(iir::LoopOrderKind::LK_Backward,
                                               Extents::VerticalLoopOrderDir::VL_InLoopOrder,
                                               false),
            Extent(1, 2));
  EXPECT_EQ(extents.getVerticalLoopOrderExtent(iir::LoopOrderKind::LK_Backward,
                                               Extents::VerticalLoopOrderDir::VL_InLoopOrder, true),
            Extent(0, 2));
}
} // anonymous namespace
