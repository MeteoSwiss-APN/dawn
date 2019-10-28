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
TEST(OffsetsTest, CartesianConstruction) {
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
TEST(OffsetsTest, UnstructuredConstruction) {
  Offsets offset1{unstructured, true, 10};
  Offsets offset2{unstructured, false, 14};

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
  auto const& cOffset = offset_cast<CartesianOffset const&>(offset.horizontalOffset());
  EXPECT_EQ(cOffset.offsetI(), 1);
  EXPECT_EQ(cOffset.offsetJ(), 2);
  EXPECT_THROW(offset_cast<UnstructuredOffset const&>(offset.horizontalOffset()), std::bad_cast);

  Offsets offset2{unstructured, true, 0};
  auto const& uOffset2 = offset_cast<UnstructuredOffset const&>(offset2.horizontalOffset());
  EXPECT_TRUE(uOffset2.hasOffset());
  EXPECT_THROW(offset_cast<CartesianOffset const&>(offset2.horizontalOffset()), std::bad_cast);

  // a default constructed offset can be cartesian or unstructured
  Offsets offset3{};
  auto const& cOffset3 = offset_cast<CartesianOffset const&>(offset3.horizontalOffset());
  EXPECT_EQ(cOffset3.offsetI(), 0);
  EXPECT_EQ(cOffset3.offsetJ(), 0);
  auto const& uOffset3 = offset_cast<UnstructuredOffset const&>(offset3.horizontalOffset());
  EXPECT_FALSE(uOffset3.hasOffset());
}

TEST(OffsetsTest, Comparison) {
  EXPECT_EQ(Offsets(cartesian, 1, 2, 3), Offsets(cartesian, 1, 2, 3));
  EXPECT_NE(Offsets(cartesian, 0, 2, 3), Offsets(cartesian, 1, 2, 3));
  EXPECT_NE(Offsets(cartesian, 1, 1, 3), Offsets(cartesian, 1, 2, 3));
  EXPECT_NE(Offsets(cartesian, 1, 2, 2), Offsets(cartesian, 1, 2, 3));
  EXPECT_EQ(Offsets(cartesian), Offsets());
  EXPECT_NE(Offsets(), Offsets(cartesian, 0, 0, 1));

  EXPECT_EQ(Offsets(unstructured), Offsets());
  EXPECT_EQ(Offsets(unstructured, false, 0), Offsets());
  EXPECT_NE(Offsets(unstructured, true, 0), Offsets());
  EXPECT_NE(Offsets(unstructured, false, 1), Offsets());

  EXPECT_EQ(Offsets(unstructured, false, 1), Offsets(unstructured, false, 1));
  EXPECT_NE(Offsets(unstructured, false, 1), Offsets(unstructured, false, 0));
  EXPECT_NE(Offsets(unstructured, true, 1), Offsets(unstructured, false, 1));

  EXPECT_THROW(Offsets(cartesian, 1, 2, 3) == Offsets(unstructured, false, 1), std::bad_cast);
  EXPECT_THROW(Offsets(cartesian) == Offsets(unstructured), std::bad_cast);
}

TEST(OffsetsTest, AddCartesian) {
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
TEST(OffsetsTest, AddUnstructured) {
  Offsets offset{unstructured, false, 4};
  offset += offset;
  EXPECT_EQ(offset, Offsets(unstructured, false, 8));

  offset += Offsets();
  EXPECT_EQ(offset, Offsets(unstructured, false, 8));

  Offsets offset2{};
  offset2 += offset;
  EXPECT_EQ(offset2, Offsets(unstructured, false, 8));

  Offsets offset3 = offset + Offsets{unstructured, true, 0};
  EXPECT_EQ(offset3, Offsets(unstructured, true, 8));

  EXPECT_THROW(Offsets(unstructured) + Offsets(cartesian), std::bad_cast);
  EXPECT_THROW(Offsets(cartesian) + Offsets(unstructured), std::bad_cast);
}

TEST(OffsetsTest, isZero) {
  EXPECT_TRUE(Offsets().isZero());
  EXPECT_TRUE(Offsets(cartesian).isZero());
  EXPECT_TRUE(Offsets(cartesian, 0, 0, 0).isZero());

  EXPECT_FALSE(Offsets(cartesian, 1, 0, 0).isZero());
  EXPECT_FALSE(Offsets(cartesian, 0, 1, 0).isZero());
  EXPECT_FALSE(Offsets(cartesian, 0, 0, 1).isZero());

  EXPECT_TRUE(Offsets(unstructured, false, 0).isZero());
  EXPECT_FALSE(Offsets(unstructured, true, 0).isZero());
  EXPECT_FALSE(Offsets(unstructured, false, 1).isZero());
}

TEST(OffsetsTest, verticalOffset) {
  EXPECT_EQ(Offsets().verticalOffset(), 0);
  EXPECT_EQ(Offsets(cartesian, 1, 2, 3).verticalOffset(), 3);
  EXPECT_EQ(Offsets(unstructured, true, 3).verticalOffset(), 3);
}

TEST(OffsetsTest, to_string) {
  // default to_string
  EXPECT_EQ(to_string(Offsets(cartesian)), "<no_horizontal_offset>, 0");
  EXPECT_EQ(to_string(Offsets(cartesian, 0, 0, 0)), "<no_horizontal_offset>, 0");
  EXPECT_EQ(to_string(Offsets(cartesian, 0, 0, 1)), "<no_horizontal_offset>, 1");
  EXPECT_EQ(to_string(Offsets(cartesian, 1, 0, 1)), "1, 0, 1");

  EXPECT_EQ(to_string(Offsets(unstructured, false, 4)), "<no_horizontal_offset>, 4");
  EXPECT_EQ(to_string(Offsets(unstructured, true, 2)), "<has_horizontal_offset>, 2");

  EXPECT_EQ(to_string(Offsets()), "<no_horizontal_offset>, 0");

  // to_string with grid type
  EXPECT_EQ(to_string(cartesian, Offsets()), "0, 0, 0");
  EXPECT_EQ(to_string(cartesian, Offsets(cartesian, 1, 2, 3)), "1, 2, 3");
  EXPECT_EQ(to_string(cartesian, Offsets(cartesian, 1, 2, 3), ";"), "1;2;3");
  auto toString1 = [](std::string const& n, int i) { return ""; };
  EXPECT_EQ(to_string(cartesian, Offsets(cartesian, 0, 2, 3), ";", toString1), "");
  auto toString2 = [](std::string const& n, int i) { return i > 0 ? n : ""; };
  EXPECT_EQ(to_string(cartesian, Offsets(cartesian, 0, 2, 3), ";", toString2), "j;k");

  EXPECT_EQ(to_string(unstructured, Offsets()), "<no_horizontal_offset>, 0");
  EXPECT_EQ(to_string(unstructured, Offsets(unstructured, false, 4)), "<no_horizontal_offset>, 4");
}

} // namespace
