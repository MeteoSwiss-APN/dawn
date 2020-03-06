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

#include "dawn/IIR/IIR.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Unittest/IIRBuilder.h"

#include <gtest/gtest.h>

using namespace dawn;

namespace {

TEST(TestFieldAccessIntervals, test_field_access_interval_01) {
  /*
    vertical_region(k_start, k_start + 12) { in = 12; }
  */
  iir::CartesianIIRBuilder b;
  auto in_f = b.field("in", iir::FieldType::ijk);

  auto instantiation = b.build(
      "Test", b.stencil(b.multistage(
                  iir::LoopOrderKind::Forward,
                  b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::Start, 0, 12,
                                     b.block(b.stmt(b.assignExpr(b.at(in_f), b.lit(12.)))))))));

  const auto& metadata = instantiation->getMetaData();
  const auto& stencils = instantiation->getStencils();
  EXPECT_EQ(stencils.size(), 1);

  int inID = metadata.getAccessIDFromName("in");
  auto fieldToIDMap = stencils[0]->getFields();
  auto field = fieldToIDMap.at(inID).field;
  EXPECT_EQ(field.getInterval(),
            (iir::Interval{sir::Interval::Start, sir::Interval::Start, 0, 12}));
}

TEST(TestFieldAccessIntervals, test_field_access_interval_02) {
  /*
    vertical_region(k_start, k_start) { in = 12; }
    vertical_region(k_end, k_end) { in = 10; }
  */
  iir::CartesianIIRBuilder b;
  auto in_f = b.field("in", iir::FieldType::ijk);
  auto instantiation = b.build(
      "Test",
      b.stencil(
          b.multistage(iir::LoopOrderKind::Forward,
                       b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::Start,
                                          b.block(b.stmt(b.assignExpr(b.at(in_f), b.lit(12.))))))),
          b.multistage(
              iir::LoopOrderKind::Forward,
              b.stage(b.doMethod(dawn::sir::Interval::End, dawn::sir::Interval::End,
                                 b.block(b.stmt(b.assignExpr(b.at(in_f), b.lit(10.)))))))));

  const auto& metadata = instantiation->getMetaData();
  const auto& stencils = instantiation->getStencils();
  EXPECT_EQ(stencils.size(), 1);

  int inID = metadata.getAccessIDFromName("in");
  auto fieldToIDMap = stencils[0]->getFields();
  auto field = fieldToIDMap.at(inID).field;
  EXPECT_EQ(field.getInterval(), (iir::Interval{sir::Interval::Start, sir::Interval::End, 0, 0}));
}

TEST(TestFieldAccessIntervals, test_field_access_interval_03) {
  /*
    vertical_region(k_start, k_start + 2) { in = 12; }
  */
  iir::CartesianIIRBuilder b;
  auto in_f = b.field("in", iir::FieldType::ijk);

  auto instantiation = b.build(
      "Test", b.stencil(b.multistage(
                  iir::LoopOrderKind::Forward,
                  b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::Start, 0, 2,
                                     b.block(b.stmt(b.assignExpr(b.at(in_f), b.lit(12.)))))))));

  const auto& metadata = instantiation->getMetaData();
  const auto& stencils = instantiation->getStencils();
  EXPECT_EQ(stencils.size(), 1);

  int inID = metadata.getAccessIDFromName("in");
  auto fieldToIDMap = stencils[0]->getFields();
  auto field = fieldToIDMap.at(inID).field;
  EXPECT_EQ(field.getInterval(), (iir::Interval{sir::Interval::Start, sir::Interval::Start, 0, 2}));
}

} // anonymous namespace
