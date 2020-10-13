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

#include "dawn/AST/LocationType.h"
#include "dawn/IIR/Field.h"
#include "dawn/SIR/SIR.h"
#include <gtest/gtest.h>

using namespace dawn;
using namespace iir;

namespace {

auto fieldDimensions = []() -> sir::FieldDimensions {
  return sir::FieldDimensions(sir::HorizontalFieldDimension(ast::cartesian, {true, true}), true);
};

TEST(TestField, Construction) {

  Field f{1,
          Field::IntendKind::Input,
          Extents(dawn::ast::cartesian, -1, 1, -2, 2, 0, 3),
          Extents(dawn::ast::cartesian, 0, 0, 0, 0, -1, 2),
          Interval{0, 2, 1, -2},
          fieldDimensions()};

  EXPECT_TRUE((f.getExtents() == Extents(dawn::ast::cartesian, -1, 1, -2, 2, -1, 3)));
  EXPECT_TRUE((f.getReadExtents() == Extents(dawn::ast::cartesian, -1, 1, -2, 2, 0, 3)));
  EXPECT_TRUE((f.getWriteExtents() == Extents(dawn::ast::cartesian, 0, 0, 0, 0, -1, 2)));
  EXPECT_TRUE((f.getInterval() == Interval{0, 2, 1, -2}));
  EXPECT_TRUE((f.computeAccessedInterval() == Interval{0, 2, 0, 1}));
  EXPECT_TRUE((f.getAccessID() == 1));
  EXPECT_TRUE((f.getIntend() == Field::IntendKind::Input));
}

TEST(TestField, Equal) {

  Field f1{1,
           Field::IntendKind::Input,
           Extents(dawn::ast::cartesian, -1, 1, -2, 2, 0, 3),
           Extents(dawn::ast::cartesian, 0, 0, 0, 0, -1, 2),
           Interval{0, 2, 1, -2},
           fieldDimensions()};

  Field f2{1,
           Field::IntendKind::Input,
           Extents(dawn::ast::cartesian, -1, 1, -1, 2, 0, 3),
           Extents(dawn::ast::cartesian, 0, 0, 0, 1, -1, 2),
           Interval{0, 2, 1, 0},
           fieldDimensions()};

  Field f3{1,
           Field::IntendKind::InputOutput,
           Extents(dawn::ast::cartesian, -1, 1, -2, 2, 0, 3),
           Extents(dawn::ast::cartesian, 0, 0, 0, 0, -1, 2),
           Interval{0, 2, 1, -2},
           fieldDimensions()};
  Field f4{2,
           Field::IntendKind::Input,
           Extents(dawn::ast::cartesian, -1, 1, -2, 2, 0, 3),
           Extents(dawn::ast::cartesian, 0, 0, 0, 0, -1, 2),
           Interval{0, 2, 1, -2},
           fieldDimensions()};

  EXPECT_TRUE((f1 == f2));
  EXPECT_TRUE((f1 != f3));
  EXPECT_TRUE((f1 != f4));
}

TEST(TestField, CartesianDimensions) {
  auto c001 =
      sir::FieldDimensions(sir::HorizontalFieldDimension(ast::cartesian, {false, false}), true);
  auto c010 =
      sir::FieldDimensions(sir::HorizontalFieldDimension(ast::cartesian, {false, true}), false);
  auto c100 =
      sir::FieldDimensions(sir::HorizontalFieldDimension(ast::cartesian, {true, false}), false);
  auto c110 =
      sir::FieldDimensions(sir::HorizontalFieldDimension(ast::cartesian, {true, true}), false);
  auto c101 =
      sir::FieldDimensions(sir::HorizontalFieldDimension(ast::cartesian, {true, false}), true);
  auto c011 =
      sir::FieldDimensions(sir::HorizontalFieldDimension(ast::cartesian, {false, true}), true);
  auto c111 =
      sir::FieldDimensions(sir::HorizontalFieldDimension(ast::cartesian, {true, true}), true);
  EXPECT_EQ(c001.numSpatialDimensions(), 1);
  EXPECT_EQ(c010.numSpatialDimensions(), 1);
  EXPECT_EQ(c100.numSpatialDimensions(), 1);
  EXPECT_EQ(c110.numSpatialDimensions(), 2);
  EXPECT_EQ(c101.numSpatialDimensions(), 2);
  EXPECT_EQ(c011.numSpatialDimensions(), 2);
  EXPECT_EQ(c111.numSpatialDimensions(), 3);
}

TEST(TestField, UnstructuredDimension) {
  auto f1d = sir::FieldDimensions(true);
  auto f2d = sir::FieldDimensions(
      sir::HorizontalFieldDimension(ast::unstructured, ast::LocationType::Cells), false);
  auto f3d = sir::FieldDimensions(
      sir::HorizontalFieldDimension(ast::unstructured, ast::LocationType::Cells), true);
  EXPECT_EQ(f1d.numSpatialDimensions(), 1);
  EXPECT_EQ(f2d.numSpatialDimensions(), 2);
  EXPECT_EQ(f3d.numSpatialDimensions(), 3);
}

TEST(TestField, Merge) {

  Field f{1,
          Field::IntendKind::Input,
          Extents(dawn::ast::cartesian, -1, 1, -2, 2, 0, 3),
          Extents(dawn::ast::cartesian, 0, 0, 0, 0, -1, 2),
          Interval{0, 2, 1, -2},
          fieldDimensions()};

  f.mergeReadExtents(Extents(dawn::ast::cartesian, -2, 1, -3, 0, 0, 0));
  EXPECT_TRUE((f.getReadExtents() == Extents(dawn::ast::cartesian, -2, 1, -3, 2, 0, 3)));
  EXPECT_TRUE((f.getWriteExtents() == Extents(dawn::ast::cartesian, 0, 0, 0, 0, -1, 2)));
  EXPECT_TRUE((f.getExtents() == Extents(dawn::ast::cartesian, -2, 1, -3, 2, -1, 3)));
} // namespace
} // anonymous namespace
