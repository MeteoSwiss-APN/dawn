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

#include "dawn/IIR/Field.h"
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
