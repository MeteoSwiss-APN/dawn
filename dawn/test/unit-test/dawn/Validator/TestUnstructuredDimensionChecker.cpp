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
      ".*Dimensions consistency check failed.*");
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
      ".*Dimensions consistency check failed.*");
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
      ".*Dimensions consistency check failed.*");
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
      ".*Dimensions consistency check failed.*");
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
      ".*Dimensions consistency check failed.*");
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
      ".*Dimensions consistency check failed.*");
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
      ".*Dimensions consistency check failed.*");
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
      ".*Dimensions consistency check failed.*");
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
      ".*Dimensions consistency check failed.*");
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
      ".*Dimensions consistency check failed.*");
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
                                             b.lit(0.), {LocType::Edges, LocType::Cells})))))))),
      ".*Dimensions consistency check failed.*");
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
                                             b.lit(0.), {LocType::Cells, LocType::Edges})))))))),
      ".*Dimensions consistency check failed.*");
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
                                          b.lit(0.), {LocType::Cells, LocType::Vertices})))))))),
      ".*Dimensions consistency check failed.*");
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
                                          b.lit(0.), {LocType::Edges, LocType::Cells})))))))),
      ".*Dimensions consistency check failed.*");
}
TEST(UnstructuredDimensionCheckerTest, StageLocType_1) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto f_c_in = b.field("f_c_in", LocType::Cells);
  auto f_c_out = b.field("f_c_out", LocType::Cells);

  auto stencil = b.build(
      "fail",
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(LocType::Edges, b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                             b.stmt(b.assignExpr(b.at(f_c_out), b.at(f_c_in))))))));
  auto result = UnstructuredDimensionChecker::checkStageLocTypeConsistency(*stencil->getIIR(),
                                                                           stencil->getMetaData());
  EXPECT_EQ(result, UnstructuredDimensionChecker::ConsistencyResult(false, dawn::SourceLocation()));
}
TEST(UnstructuredDimensionCheckerTest, StageLocType_2) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto f_c_in = b.field("f_c_in", LocType::Cells);
  auto f_c_out = b.field("f_c_out", LocType::Cells);
  auto f_e_in = b.field("f_e_in", LocType::Edges);
  auto f_e_out = b.field("f_e_out", LocType::Edges);

  auto stencil = b.build(
      "fail",
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(LocType::Cells, b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                             b.stmt(b.assignExpr(b.at(f_c_out), b.at(f_c_in))),
                                             b.stmt(b.assignExpr(b.at(f_e_out), b.at(f_e_in))))))));
  auto result = UnstructuredDimensionChecker::checkStageLocTypeConsistency(*stencil->getIIR(),
                                                                           stencil->getMetaData());
  EXPECT_EQ(result, UnstructuredDimensionChecker::ConsistencyResult(false, dawn::SourceLocation()));
}
TEST(UnstructuredDimensionCheckerTest, StageLocType_3) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto f_v_in = b.field("f_v_in", LocType::Vertices);
  auto f_v_out = b.field("f_v_out", LocType::Vertices);

  auto stencil =
      b.build("pass", b.stencil(b.multistage(
                          LoopOrderKind::Parallel,
                          b.stage(LocType::Vertices,
                                  b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                             b.stmt(b.assignExpr(b.at(f_v_out), b.at(f_v_in))))))));
  auto result = UnstructuredDimensionChecker::checkStageLocTypeConsistency(*stencil->getIIR(),
                                                                           stencil->getMetaData());
  EXPECT_EQ(result, UnstructuredDimensionChecker::ConsistencyResult(true, dawn::SourceLocation()));
}
} // namespace