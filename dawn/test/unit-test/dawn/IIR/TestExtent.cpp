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

TEST(ExtentsTest, CartesianConstruction) {
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
TEST(ExtentsTest, UnstructuredConstruction) {
  Extents expected(ast::unstructured, true, Extent(1, 1));

  Extents extents = expected;
  EXPECT_EQ(extents, expected);

  Extents extents2 = std::move(extents);
  EXPECT_EQ(extents2, expected);

  Extents extents3{extents2};
  EXPECT_EQ(extents2, expected);
  EXPECT_EQ(extents3, expected);

  Extents extents4{std::move(extents3)};
  EXPECT_EQ(extents4, expected);

  EXPECT_EQ(Extents(ast::Offsets{ast::unstructured, true, 1}),
            Extents(ast::unstructured, true, Extent{1, 1}));
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
  EXPECT_EQ(extents2, Extents());
}

TEST(ExtentsTest, ExtentCast) {
  Extents extent1{ast::cartesian, -1, 1, -2, 2, -3, 3};
  auto const& cExtent = extent_cast<CartesianExtent const&>(extent1.horizontalExtent());
  EXPECT_EQ(cExtent.iMinus(), -1);
  EXPECT_EQ(cExtent.iPlus(), 1);
  EXPECT_EQ(cExtent.jMinus(), -2);
  EXPECT_EQ(cExtent.jPlus(), 2);
  EXPECT_THROW(extent_cast<UnstructuredExtent const&>(extent1.horizontalExtent()), std::bad_cast);

  Extents extent2{ast::unstructured, true, Extent{-3, 3}};
  auto const& uExtent2 = extent_cast<UnstructuredExtent const&>(extent2.horizontalExtent());
  EXPECT_TRUE(uExtent2.hasExtent());
  EXPECT_THROW(extent_cast<CartesianExtent const&>(extent2.horizontalExtent()), std::bad_cast);

  Extents extent3;
  auto const& cExtent3 = extent_cast<CartesianExtent const&>(extent3.horizontalExtent());
  EXPECT_EQ(cExtent3.iMinus(), 0);
  EXPECT_EQ(cExtent3.iPlus(), 0);
  EXPECT_EQ(cExtent3.jMinus(), 0);
  EXPECT_EQ(cExtent3.jPlus(), 0);
  auto const& uExtent3 = extent_cast<UnstructuredExtent const&>(extent3.horizontalExtent());
  EXPECT_FALSE(uExtent3.hasExtent());
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

  EXPECT_TRUE(Extents(ast::unstructured, false, Extent{0, 0}).isHorizontalPointwise());
  EXPECT_FALSE(Extents(ast::unstructured, true, Extent{0, 0}).isHorizontalPointwise());

  EXPECT_TRUE(Extents().isPointwise());
}

TEST(ExtentsTest, MergeWithExtent) {
  Extents extents(ast::Offsets{ast::cartesian, -1, 1, 0});
  Extents extentsToMerge(ast::Offsets{ast::cartesian, 3, 2, 1});
  EXPECT_EQ(merge(extents, extentsToMerge), Extents(ast::cartesian, -1, 3, 1, 2, 0, 1));

  EXPECT_EQ(merge(Extents(), Extents()), Extents());
  EXPECT_EQ(merge(Extents(ast::cartesian, -1, -1, 1, 1, 0, 0), Extents()),
            Extents(ast::cartesian, -1, 0, 0, 1, 0, 0));
  EXPECT_EQ(merge(Extents(ast::cartesian, -1, 2, -2, 1, 0, 0),
                  Extents(ast::cartesian, -2, 1, -1, 2, 0, 0)),
            Extents(ast::cartesian, -2, 2, -2, 2, 0, 0));

  EXPECT_EQ(merge(Extents(ast::unstructured, false, Extent(0, 1)), Extents()),
            Extents(ast::unstructured, false, Extent(0, 1)));
  EXPECT_EQ(merge(Extents(ast::unstructured, true, Extent(0, 1)), Extents()),
            Extents(ast::unstructured, true, Extent(0, 1)));
  EXPECT_EQ(merge(Extents(ast::unstructured, false, Extent(0, 1)),
                  Extents(ast::unstructured, true, Extent(-1, 0))),
            Extents(ast::unstructured, true, Extent(-1, 1)));

  EXPECT_THROW(merge(Extents(ast::unstructured), Extents(ast::cartesian)), std::bad_cast);
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
TEST(ExtentsTest, Limit) {
  EXPECT_EQ(limit(Extents{ast::cartesian, -2, 2, -2, 2, 0, 0},
                  Extents{ast::cartesian, -1, 1, -1, 1, 0, 0}),
            Extents(ast::cartesian, -1, 1, -1, 1, 0, 0));

  EXPECT_EQ(limit(Extents{ast::cartesian, -2, 1, -1, 2, 0, 0},
                  Extents{ast::cartesian, -1, 2, -2, 1, 0, 0}),
            Extents(ast::cartesian, -1, 1, -1, 1, 0, 0));

  EXPECT_EQ(limit(Extents{ast::unstructured, true, Extent{-3, 5}},
                  Extents{ast::unstructured, false, Extent{-1, 8}}),
            Extents(ast::unstructured, false, Extent{-1, 5}));

  EXPECT_THROW(limit(Extents{ast::unstructured}, Extents{ast::cartesian}), std::bad_cast);
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

  EXPECT_EQ(Extents(ast::unstructured, false, Extent(0, 3)) +
                Extents(ast::unstructured, false, Extent(1, 2)),
            Extents(ast::unstructured, false, Extent(1, 5)));
  EXPECT_EQ(Extents(ast::unstructured, true, Extent(0, 3)) +
                Extents(ast::unstructured, false, Extent(-1, 2)),
            Extents(ast::unstructured, true, Extent(-1, 5)));
  EXPECT_EQ(Extents(ast::unstructured, false, Extent(0, 3)) +
                Extents(ast::unstructured, false, Extent(-1, 2)),
            Extents(ast::unstructured, false, Extent(-1, 5)));
}

TEST(ExtentsTest, Stringify) {
  EXPECT_EQ(to_string(Extents()), "[<no_horizontal_extent>, (0, 0)]");

  EXPECT_EQ(to_string(Extents(ast::cartesian)), "[<no_horizontal_extent>, (0, 0)]");
  EXPECT_EQ(to_string(Extents(ast::cartesian, 0, 0, 0, 0, -1, 1)),
            "[<no_horizontal_extent>, (-1, 1)]");
  EXPECT_EQ(to_string(Extents(ast::cartesian, -1, 1, -2, 2, -3, 3)), "[(-1, 1), (-2, 2), (-3, 3)]");

  EXPECT_EQ(to_string(Extents(ast::unstructured)), "[<no_horizontal_extent>, (0, 0)]");
  EXPECT_EQ(to_string(Extents(ast::unstructured, false, Extent(-1, 3))),
            "[<no_horizontal_extent>, (-1, 3)]");
  EXPECT_EQ(to_string(Extents(ast::unstructured, true, Extent(0, 0))),
            "[<has_horizontal_extent>, (0, 0)]");
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
