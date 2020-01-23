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
#include "dawn/Validator/UnstructuredDimensionChecker.h"
#include <gtest/gtest.h>

using namespace dawn;

namespace {

TEST(UnstructuredDimensionCheckerTest, AssignmentCase0_0) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto sparse_field1 =
      b.field("sparse_field1", {LocType::Edges, LocType::Cells, LocType::Vertices});
  auto sparse_field2 = b.field("sparse_field2", {LocType::Edges, LocType::Cells, LocType::Cells});

  EXPECT_DEATH(
      b.build("fail", b.stencil(b.multistage(
                          LoopOrderKind::Parallel,
                          b.stage(b.doMethod(
                              dawn::sir::Interval::Start, dawn::sir::Interval::End,
                              b.stmt(b.assignExpr(b.at(sparse_field1), b.at(sparse_field2)))))))),
      ".*checkDimensionsConsistency.*");
}
TEST(UnstructuredDimensionCheckerTest, AssignmentCase0_1) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto sparse_field1 = b.field("sparse_field1", {LocType::Edges, LocType::Cells});
  auto sparse_field2 = b.field("sparse_field2", {LocType::Edges, LocType::Cells, LocType::Cells});

  EXPECT_DEATH(
      b.build("fail", b.stencil(b.multistage(
                          LoopOrderKind::Parallel,
                          b.stage(b.doMethod(
                              dawn::sir::Interval::Start, dawn::sir::Interval::End,
                              b.stmt(b.assignExpr(b.at(sparse_field1), b.at(sparse_field2)))))))),
      ".*checkDimensionsConsistency.*");
}
TEST(UnstructuredDimensionCheckerTest, AssignmentCase1_0) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto sparse_field = b.field("sparse_field", {LocType::Edges, LocType::Cells, LocType::Vertices});
  auto dense_field = b.field("dense_field", LocType::Cells);

  EXPECT_DEATH(
      b.build(
          "fail",
          b.stencil(b.multistage(
              LoopOrderKind::Parallel,
              b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                 b.stmt(b.assignExpr(b.at(sparse_field), b.at(dense_field)))))))),
      ".*checkDimensionsConsistency.*");
}
TEST(UnstructuredDimensionCheckerTest, AssignmentCase2_0) {
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
TEST(UnstructuredDimensionCheckerTest, AssignmentNoCase_0) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto dense_field = b.field("dense_field", LocType::Vertices);
  auto sparse_field = b.field("sparse_field", {LocType::Edges, LocType::Cells, LocType::Vertices});

  EXPECT_DEATH(
      b.build(
          "fail",
          b.stencil(b.multistage(
              LoopOrderKind::Parallel,
              b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                 b.stmt(b.assignExpr(b.at(dense_field), b.at(sparse_field)))))))),
      ".*checkDimensionsConsistency.*");
}
TEST(UnstructuredDimensionCheckerTest, BinaryOpCase0_0) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto sparse_field1 =
      b.field("sparse_field1", {LocType::Edges, LocType::Cells, LocType::Vertices});
  auto sparse_field2 = b.field("sparse_field2", {LocType::Edges, LocType::Cells, LocType::Cells});

  EXPECT_DEATH(
      b.build("fail",
              b.stencil(b.multistage(
                  LoopOrderKind::Parallel,
                  b.stage(b.doMethod(
                      dawn::sir::Interval::Start, dawn::sir::Interval::End,
                      b.stmt(b.binaryExpr(b.at(sparse_field1), b.at(sparse_field2), Op::plus))))))),
      ".*checkDimensionsConsistency.*");
}
TEST(UnstructuredDimensionCheckerTest, BinaryOpCase0_1) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto sparse_field1 = b.field("sparse_field1", {LocType::Edges, LocType::Cells});
  auto sparse_field2 = b.field("sparse_field2", {LocType::Edges, LocType::Cells, LocType::Cells});

  EXPECT_DEATH(
      b.build("fail",
              b.stencil(b.multistage(
                  LoopOrderKind::Parallel,
                  b.stage(b.doMethod(
                      dawn::sir::Interval::Start, dawn::sir::Interval::End,
                      b.stmt(b.binaryExpr(b.at(sparse_field1), b.at(sparse_field2), Op::plus))))))),
      ".*checkDimensionsConsistency.*");
}
TEST(UnstructuredDimensionCheckerTest, BinaryOpCase1_0) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto dense_field = b.field("dense_field", LocType::Cells);
  auto sparse_field = b.field("sparse_field", {LocType::Edges, LocType::Cells, LocType::Vertices});

  EXPECT_DEATH(
      b.build("fail",
              b.stencil(b.multistage(
                  LoopOrderKind::Parallel,
                  b.stage(b.doMethod(
                      dawn::sir::Interval::Start, dawn::sir::Interval::End,
                      b.stmt(b.binaryExpr(b.at(dense_field), b.at(sparse_field), Op::plus))))))),
      ".*checkDimensionsConsistency.*");
}
TEST(UnstructuredDimensionCheckerTest, BinaryOpCase2_0) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto sparse_field = b.field("sparse_field", {LocType::Edges, LocType::Cells, LocType::Vertices});
  auto dense_field = b.field("dense_field", LocType::Cells);

  EXPECT_DEATH(
      b.build("fail", b.stencil(b.multistage(
                          LoopOrderKind::Parallel,
                          b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                             b.stmt(b.binaryExpr(b.at(sparse_field),
                                                                 b.at(dense_field), Op::plus))))))),
      ".*checkDimensionsConsistency.*");
}
TEST(UnstructuredDimensionCheckerTest, BinaryOpCase3_0) {
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
TEST(UnstructuredDimensionCheckerTest, ReduceDense_0) {
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
TEST(UnstructuredDimensionCheckerTest, ReduceDense_1) {
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
TEST(UnstructuredDimensionCheckerTest, ReduceSparse_0) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto sparse_field = b.field("edge_field", {LocType::Edges, LocType::Cells, LocType::Vertices});
  auto cell_field = b.field("cell_field", LocType::Cells);

  EXPECT_DEATH(
      b.build(
          "fail",
          b.stencil(b.multistage(
              LoopOrderKind::Parallel,
              b.stage(b.doMethod(
                  dawn::sir::Interval::Start, dawn::sir::Interval::End,
                  b.stmt(b.assignExpr(b.at(cell_field),
                                      b.reduceOverNeighborExpr(
                                          Op::plus, b.at(sparse_field, HOffsetType::withOffset, 0),
                                          b.lit(0.), LocType::Cells, LocType::Vertices)))))))),
      ".*checkDimensionsConsistency.*");
}
TEST(UnstructuredDimensionCheckerTest, ReduceSparse_1) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto sparse_field = b.field("edge_field", {LocType::Edges, LocType::Cells, LocType::Vertices});
  auto edge_field = b.field("cell_field", LocType::Edges);

  EXPECT_DEATH(
      b.build(
          "fail",
          b.stencil(b.multistage(
              LoopOrderKind::Parallel,
              b.stage(b.doMethod(
                  dawn::sir::Interval::Start, dawn::sir::Interval::End,
                  b.stmt(b.assignExpr(b.at(edge_field),
                                      b.reduceOverNeighborExpr(
                                          Op::plus, b.at(sparse_field, HOffsetType::withOffset, 0),
                                          b.lit(0.), LocType::Edges, LocType::Cells)))))))),
      ".*checkDimensionsConsistency.*");
}
} // namespace