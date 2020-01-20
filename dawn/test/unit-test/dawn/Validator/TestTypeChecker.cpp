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

#include "dawn/Unittest/IIRBuilder.h"
#include "dawn/Validator/TypeChecker.h"
#include <gtest/gtest.h>

using namespace dawn;

namespace {

TEST(TypeCheckerTest, UnstructuredCheck01) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto cell_f = b.field("cell_field", LocType::Cells);
  auto edge_f = b.field("edge_field", LocType::Edges);

  EXPECT_DEATH(
      b.build("fail", b.stencil(b.multistage(
                          LoopOrderKind::Parallel,
                          b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                             b.stmt(b.assignExpr(b.at(cell_f), b.at(edge_f)))))))),
      ".*checkDimensionsConsistency.*");
}
TEST(TypeCheckerTest, UnstructuredCheck02) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto cell_f = b.field("cell_field", LocType::Cells);
  auto edge_f = b.field("edge_field", LocType::Edges);

  EXPECT_DEATH(
      b.build(
          "fail",
          b.stencil(b.multistage(
              LoopOrderKind::Parallel,
              b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                 b.stmt(b.binaryExpr(b.at(cell_f), b.at(edge_f), Op::plus))))))),
      ".*checkDimensionsConsistency.*");
}
TEST(TypeCheckerTest, UnstructuredCheck03) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto edge_field = b.field("edge_field", LocType::Edges);
  auto cell_field = b.field("cell_field", LocType::Cells);

  EXPECT_DEATH(
      b.build("fail",
              b.stencil(b.multistage(
                  LoopOrderKind::Parallel,
                  b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                     b.stmt(b.assignExpr(
                                         b.at(cell_field),
                                         b.reduceOverNeighborExpr(
                                             Op::plus, b.at(edge_field, HOffsetType::withOffset, 0),
                                             b.lit(0.), LocType::Edges, LocType::Cells)))))))),
      ".*checkDimensionsConsistency.*");
}
TEST(TypeCheckerTest, UnstructuredCheck04) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto edge_field = b.field("edge_field", LocType::Edges);
  auto cell_field = b.field("cell_field", LocType::Cells);

  EXPECT_DEATH(
      b.build("fail",
              b.stencil(b.multistage(
                  LoopOrderKind::Parallel,
                  b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                     b.stmt(b.assignExpr(
                                         b.at(edge_field),
                                         b.reduceOverNeighborExpr(
                                             Op::plus, b.at(cell_field, HOffsetType::withOffset, 0),
                                             b.lit(0.), LocType::Cells, LocType::Edges)))))))),
      ".*checkDimensionsConsistency.*");
}
} // namespace