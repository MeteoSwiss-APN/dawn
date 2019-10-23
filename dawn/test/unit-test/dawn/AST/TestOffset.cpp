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
using namespace ast;

namespace {
TEST(OffsetsTest, Construction) {
  EXPECT_EQ(Offsets(cartesian, 1, 2, 3), Offsets(cartesian, std::array<int, 3>{1, 2, 3}));
  EXPECT_EQ(Offsets(cartesian, 0, 0, 0), Offsets(cartesian));
  EXPECT_EQ(Offsets(cartesian, 0, 0, 0), Offsets());

  Offsets offset1{cartesian, 1, 2, 3};
  Offsets offset2{cartesian, 3, 2, 1};

  // copy assignment
  offset2 = offset1;
  EXPECT_EQ(offset1, offset2);

  // move assignment
  Offsets offset3;
  offset3 = std::move(offset2);
  EXPECT_EQ(offset1, offset3);

  // move constructor
  Offsets offset4{std::move(offset3)};
  EXPECT_EQ(offset1, offset4);

  // copy constructor
  Offsets offset5{offset1};
  EXPECT_EQ(offset1, offset5);
}

TEST(OffsetsTest, NullConstruction) {
  Offsets offset1{cartesian, 1, 2, 3};
  Offsets offset2{cartesian, 3, 2, 1};

  Offsets nullOffset1;
  Offsets nullOffset2;

  // copy assignment
  nullOffset1 = offset1;
  EXPECT_EQ(nullOffset1, offset1);
  nullOffset1 = nullOffset2;
  EXPECT_EQ(nullOffset1, nullOffset2);

  // move assignment
  Offsets offset3{offset1};
  offset3 = std::move(nullOffset2);
  EXPECT_EQ(nullOffset1, offset3);
  nullOffset2 = Offsets{};

  // move constructor
  Offsets offset4{std::move(nullOffset2)};
  EXPECT_EQ(nullOffset1, offset4);
  nullOffset2 = Offsets{};

  // copy constructor
  Offsets offset5{nullOffset2};
  EXPECT_EQ(nullOffset1, nullOffset2);
  EXPECT_EQ(nullOffset1, offset5);
}

TEST(OffsetsTest, OffsetCast) {
  Offsets offset{cartesian, 1, 2, 3};
  auto const& hOffset = offset_cast<CartesianOffset const&>(offset.horizontalOffset());
  EXPECT_EQ(hOffset.offsetI(), 1);
  EXPECT_EQ(hOffset.offsetJ(), 2);

  Offsets offset2{};
  auto const& hOffset2 = offset_cast<CartesianOffset const&>(offset2.horizontalOffset());
  EXPECT_EQ(hOffset2.offsetI(), 0);
  EXPECT_EQ(hOffset2.offsetJ(), 0);
}

TEST(OffsetsTest, Add) {
  Offsets offset{cartesian, 1, 2, 3};
  offset += offset;
  EXPECT_EQ(offset, Offsets(cartesian, 2, 4, 6));

  offset += Offsets();
  EXPECT_EQ(offset, Offsets(cartesian, 2, 4, 6));

  Offsets offset2{};
  offset2 += offset2;
  EXPECT_EQ(offset2, Offsets(cartesian));
  offset2 += offset;
  EXPECT_EQ(offset2, Offsets(cartesian, 2, 4, 6));
}

TEST(OffsetsTest, isZero) {
  EXPECT_TRUE(Offsets().isZero());
  EXPECT_TRUE(Offsets(cartesian).isZero());
  EXPECT_TRUE(Offsets(cartesian, 0, 0, 0).isZero());

  EXPECT_FALSE(Offsets(cartesian, 1, 0, 0).isZero());
  EXPECT_FALSE(Offsets(cartesian, 0, 1, 0).isZero());
  EXPECT_FALSE(Offsets(cartesian, 0, 0, 1).isZero());
}

TEST(OffsetsTest, verticalOffset) {
  EXPECT_EQ(Offsets().verticalOffset(), 0);
  EXPECT_EQ(Offsets(cartesian, 1, 2, 3).verticalOffset(), 3);
}

} // namespace
