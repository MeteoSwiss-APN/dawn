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
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Support/Exception.h"
#include "dawn/Unittest/IIRBuilder.h"
#include "dawn/Validator/UnstructuredDimensionChecker.h"
#include <gtest/gtest.h>
#include <memory>

using namespace dawn;

namespace {

TEST(UnstructuredDimensionCheckerTest, AssignmentCase0) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto sparse_field1 =
      b.field("sparse_field1", {LocType::Edges, LocType::Cells, LocType::Vertices});
  auto sparse_field2 = b.field("sparse_field2", {LocType::Edges, LocType::Cells, LocType::Edges});

  EXPECT_DEATH(
      b.build(
          "fail",
          b.stencil(b.multistage(
              LoopOrderKind::Parallel,
              b.stage(b.doMethod(
                  dawn::ast::Interval::Start, dawn::ast::Interval::End,
                  b.loopStmtChain(b.stmt(b.assignExpr(b.at(sparse_field1), b.at(sparse_field2))),
                                  {LocType::Edges, LocType::Cells, LocType::Vertices})))))),
      ".*Dimensions consistency check failed.*");
}
TEST(UnstructuredDimensionCheckerTest, AssignmentCase1) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto sparse_field1 = b.field("sparse_field1", {LocType::Edges, LocType::Cells});
  auto sparse_field2 =
      b.field("sparse_field2", {LocType::Edges, LocType::Cells, LocType::Edges, LocType::Cells});

  EXPECT_DEATH(
      b.build("fail",
              b.stencil(b.multistage(
                  LoopOrderKind::Parallel,
                  b.stage(b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                                     b.loopStmtChain(b.stmt(b.assignExpr(b.at(sparse_field1),
                                                                         b.at(sparse_field2))),
                                                     {LocType::Edges, LocType::Cells})))))),
      ".*Dimensions consistency check failed.*");
}
TEST(UnstructuredDimensionCheckerTest, AssignmentCase2) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto sparse_field = b.field("sparse_field", {LocType::Edges, LocType::Cells, LocType::Vertices});
  auto dense_field = b.field("dense_field", LocType::Cells);

  EXPECT_DEATH(
      b.build("fail",
              b.stencil(b.multistage(
                  LoopOrderKind::Parallel,
                  b.stage(b.doMethod(
                      dawn::ast::Interval::Start, dawn::ast::Interval::End,
                      b.loopStmtChain(b.stmt(b.assignExpr(b.at(sparse_field), b.at(dense_field))),
                                      {ast::LocationType::Edges, ast::LocationType::Cells,
                                       ast::LocationType::Vertices})))))),
      ".*Dimensions consistency check failed.*");
}
TEST(UnstructuredDimensionCheckerTest, AssignmentCase3) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto cell_f = b.field("cell_field", LocType::Cells);
  auto edge_f = b.field("edge_field", LocType::Edges);

  EXPECT_DEATH(
      b.build("fail", b.stencil(b.multistage(
                          LoopOrderKind::Parallel,
                          b.stage(b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                                             b.stmt(b.assignExpr(b.at(cell_f), b.at(edge_f)))))))),
      ".*Dimensions consistency check failed.*");
}
TEST(UnstructuredDimensionCheckerTest, AssignmentCase4) {
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
              b.stage(b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                                 b.stmt(b.assignExpr(b.at(dense_field), b.at(sparse_field)))))))),
      ".*Dimensions consistency check failed.*");
}
TEST(UnstructuredDimensionCheckerTest, AssignmentCase5) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto dense_field_v = b.field("dense_v_field", LocType::Vertices);
  auto dense_field_e = b.field("dense_e_field", LocType::Edges);
  auto sparse_field = b.field("sparse_field", {LocType::Edges, LocType::Cells});

  auto stencil = b.build(
      "passing",
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(b.doMethod(
              dawn::ast::Interval::Start, dawn::ast::Interval::End,
              b.loopStmtChain(
                  b.stmt(b.assignExpr(
                      b.at(sparse_field),
                      b.binaryExpr(b.reduceOverNeighborExpr(Op::plus, b.at(dense_field_e),
                                                            b.lit(0.),
                                                            {LocType::Cells, LocType::Edges}),
                                   b.at(sparse_field, HOffsetType::withOffset, 0), Op::plus))),
                  {LocType::Edges, LocType::Cells}))))));

  auto result = UnstructuredDimensionChecker::checkDimensionsConsistency(*stencil->getIIR(),
                                                                         stencil->getMetaData());
  EXPECT_EQ(result, UnstructuredDimensionChecker::ConsistencyResult(true, dawn::SourceLocation()));
}
TEST(UnstructuredDimensionCheckerTest, BinaryOpCase0) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto sparse_field1 =
      b.field("sparse_field1", {LocType::Edges, LocType::Cells, LocType::Vertices});
  auto sparse_field2 =
      b.field("sparse_field2", {LocType::Edges, LocType::Cells, LocType::Edges, LocType::Cells});

  EXPECT_DEATH(
      b.build("fail",
              b.stencil(b.multistage(
                  LoopOrderKind::Parallel,
                  b.stage(b.doMethod(
                      dawn::ast::Interval::Start, dawn::ast::Interval::End,
                      b.stmt(b.binaryExpr(b.at(sparse_field1), b.at(sparse_field2), Op::plus))))))),
      ".*Dimensions consistency check failed.*");
}
TEST(UnstructuredDimensionCheckerTest, BinaryOpCase1) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto sparse_field1 = b.field("sparse_field1", {LocType::Edges, LocType::Cells});
  auto sparse_field2 =
      b.field("sparse_field2", {LocType::Edges, LocType::Cells, LocType::Edges, LocType::Cells});

  EXPECT_DEATH(
      b.build("fail",
              b.stencil(b.multistage(
                  LoopOrderKind::Parallel,
                  b.stage(b.doMethod(
                      dawn::ast::Interval::Start, dawn::ast::Interval::End,
                      b.stmt(b.binaryExpr(b.at(sparse_field1), b.at(sparse_field2), Op::plus))))))),
      ".*Dimensions consistency check failed.*");
}
TEST(UnstructuredDimensionCheckerTest, BinaryOpCase2) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto out_field = b.field("out_field", LocType::Edges);
  auto dense_field = b.field("dense_field", LocType::Cells);
  auto sparse_field = b.field("sparse_field", {LocType::Edges, LocType::Cells, LocType::Vertices});

  EXPECT_DEATH(
      b.build(
          "fail",
          b.stencil(b.multistage(
              LoopOrderKind::Parallel,
              b.stage(b.doMethod(
                  dawn::ast::Interval::Start, dawn::ast::Interval::End,
                  b.stmt(b.assignExpr(
                      b.at(out_field),
                      b.reduceOverNeighborExpr(
                          Op::plus, b.binaryExpr(b.at(dense_field), b.at(sparse_field), Op::plus),
                          b.lit(0.0), {LocType::Edges, LocType::Cells, LocType::Vertices})))))))),
      ".*Dimensions consistency check failed.*");
}
TEST(UnstructuredDimensionCheckerTest, BinaryOpCase3) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto dense_field = b.field("dense_field", LocType::Cells);
  auto sparse_field = b.field("sparse_field", {LocType::Cells, LocType::Edges, LocType::Vertices});

  auto stencil = b.build(
      "pass",
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(b.doMethod(
              dawn::ast::Interval::Start, dawn::ast::Interval::End,
              b.stmt(b.assignExpr(
                  b.at(dense_field),
                  b.reduceOverNeighborExpr(
                      Op::plus, b.binaryExpr(b.at(dense_field), b.at(sparse_field), Op::plus),
                      b.lit(0.0), {LocType::Cells, LocType::Edges, LocType::Vertices}))))))));
  auto result = UnstructuredDimensionChecker::checkDimensionsConsistency(*stencil->getIIR(),
                                                                         stencil->getMetaData());
  EXPECT_EQ(result, UnstructuredDimensionChecker::ConsistencyResult(true, dawn::SourceLocation()));
}
TEST(UnstructuredDimensionCheckerTest, BinaryOpCase4) {
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
              b.stage(b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                                 b.stmt(b.binaryExpr(b.at(cell_f), b.at(edge_f), Op::plus))))))),
      ".*Dimensions consistency check failed.*");
}
TEST(UnstructuredDimensionCheckerTest, ReduceDense0) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto edge_field = b.field("edge_field", LocType::Edges);
  auto cell_field = b.field("cell_field", LocType::Cells);

  EXPECT_DEATH(
      b.build("fail",
              b.stencil(b.multistage(
                  LoopOrderKind::Parallel,
                  b.stage(b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                                     b.stmt(b.assignExpr(
                                         b.at(cell_field),
                                         b.reduceOverNeighborExpr(
                                             Op::plus, b.at(edge_field, HOffsetType::withOffset, 0),
                                             b.lit(0.), {LocType::Edges, LocType::Cells})))))))),
      ".*Dimensions consistency check failed.*");
}
TEST(UnstructuredDimensionCheckerTest, ReduceDense1) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto edge_field = b.field("edge_field", LocType::Edges);
  auto cell_field = b.field("cell_field", LocType::Cells);

  EXPECT_DEATH(
      b.build("fail",
              b.stencil(b.multistage(
                  LoopOrderKind::Parallel,
                  b.stage(b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                                     b.stmt(b.assignExpr(
                                         b.at(edge_field),
                                         b.reduceOverNeighborExpr(
                                             Op::plus, b.at(cell_field, HOffsetType::withOffset, 0),
                                             b.lit(0.), {LocType::Cells, LocType::Edges})))))))),
      ".*Dimensions consistency check failed.*");
}
TEST(UnstructuredDimensionCheckerTest, ReduceDense_2) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto edge_field = b.field("edge_field", LocType::Edges);
  auto cell_field = b.field("cell_field", LocType::Cells);
  auto node_field = b.field("node_field", LocType::Vertices);

  EXPECT_DEATH(
      b.build("fail",
              b.stencil(b.multistage(
                  LoopOrderKind::Parallel,
                  b.stage(b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                                     b.stmt(b.assignExpr(
                                         b.at(edge_field),
                                         b.reduceOverNeighborExpr(
                                             Op::plus, b.at(cell_field, HOffsetType::withOffset, 0),
                                             b.lit(0.), {LocType::Cells, LocType::Edges},
                                             {b.at(node_field), b.at(node_field)})))))))),
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
                  dawn::ast::Interval::Start, dawn::ast::Interval::End,
                  b.stmt(b.assignExpr(b.at(cell_field),
                                      b.reduceOverNeighborExpr(
                                          Op::plus, b.at(sparse_field, HOffsetType::withOffset, 0),
                                          b.lit(0.), {LocType::Cells, LocType::Vertices})))))))),
      ".*Dimensions consistency check failed.*");
}
TEST(UnstructuredDimensionCheckerTest, ReduceSparse1) {
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
                  dawn::ast::Interval::Start, dawn::ast::Interval::End,
                  b.stmt(b.assignExpr(b.at(edge_field),
                                      b.reduceOverNeighborExpr(
                                          Op::plus, b.at(sparse_field, HOffsetType::withOffset, 0),
                                          b.lit(0.), {LocType::Edges, LocType::Cells})))))))),
      ".*Dimensions consistency check failed.*");
}

TEST(UnstructuredDimensionCheckerTest, Loop0) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto sparse_f = b.field("sparse", {LocType::Edges, LocType::Cells, LocType::Vertices});
  auto e_f = b.field("e", LocType::Edges);
  auto v_f = b.field("v", LocType::Vertices);

  auto stencil = b.build(
      "passing",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Parallel,
          b.stage(LocType::Edges,
                  b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                             b.loopStmtChain(
                                 b.stmt(b.assignExpr(
                                     b.at(sparse_f),
                                     b.binaryExpr(b.at(e_f), b.at(v_f, HOffsetType::withOffset, 0),
                                                  Op::plus))),
                                 {LocType::Edges, LocType::Cells, LocType::Vertices}))))));
  auto result = UnstructuredDimensionChecker::checkDimensionsConsistency(*stencil->getIIR(),
                                                                         stencil->getMetaData());
  EXPECT_EQ(result, UnstructuredDimensionChecker::ConsistencyResult(true, dawn::SourceLocation()));
}

TEST(UnstructuredDimensionCheckerTest, Loop1) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto sparse_f = b.field("sparse", {LocType::Edges, LocType::Cells, LocType::Vertices});
  auto e_f = b.field("e", LocType::Edges);
  auto v_f = b.field("v", LocType::Vertices);

  EXPECT_DEATH(
      b.build("fail",
              b.stencil(b.multistage(
                  dawn::iir::LoopOrderKind::Parallel,
                  b.stage(LocType::Edges,
                          b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                                     b.loopStmtChain(
                                         b.stmt(b.assignExpr(
                                             b.at(sparse_f),
                                             b.binaryExpr(b.at(e_f, HOffsetType::withOffset, 0),
                                                          b.at(v_f, HOffsetType::withOffset, 0),
                                                          Op::plus))),
                                         {LocType::Edges, LocType::Cells, LocType::Vertices})))))),
      ".*Dimensions consistency check failed.*");
}

TEST(UnstructuredDimensionCheckerTest, NestedReduce0) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto sparse_f = b.field("sparse", {LocType::Edges, LocType::Cells, LocType::Vertices});
  auto e_f = b.field("e", LocType::Edges);
  auto c_f = b.field("c", LocType::Cells);

  auto stencil = b.build(
      "pass",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Parallel,
          b.stage(LocType::Cells,
                  b.doMethod(
                      dawn::ast::Interval::Start, dawn::ast::Interval::End,
                      b.loopStmtChain(b.stmt(b.assignExpr(
                                          b.at(sparse_f),
                                          b.reduceOverNeighborExpr(
                                              Op::plus,
                                              b.binaryExpr(b.at(e_f),
                                                           b.reduceOverNeighborExpr(
                                                               Op::plus, b.at(c_f), b.lit(0.),
                                                               {LocType::Vertices, LocType::Cells}),
                                                           Op::multiply),
                                              b.lit(0.), {LocType::Edges, LocType::Vertices}))),
                                      {LocType::Edges, LocType::Cells, LocType::Vertices}))))));
  auto result = UnstructuredDimensionChecker::checkDimensionsConsistency(*stencil->getIIR(),
                                                                         stencil->getMetaData());
  EXPECT_EQ(result, UnstructuredDimensionChecker::ConsistencyResult(true, dawn::SourceLocation()));
}

TEST(UnstructuredDimensionCheckerTest, NestedReduce1) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto sparse_f = b.field("sparse", {LocType::Edges, LocType::Cells, LocType::Vertices});
  auto e_f = b.field("e", LocType::Edges);
  auto c_f = b.field("c", LocType::Cells);

  EXPECT_DEATH(
      b.build(
          "fail",
          b.stencil(b.multistage(
              dawn::iir::LoopOrderKind::Parallel,
              b.stage(LocType::Cells,
                      b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                                 b.loopStmtChain(
                                     b.stmt(b.assignExpr(
                                         b.at(sparse_f),
                                         b.reduceOverNeighborExpr(
                                             Op::plus,
                                             b.binaryExpr(b.at(e_f),
                                                          b.reduceOverNeighborExpr(
                                                              Op::plus, b.at(c_f), b.lit(0.),
                                                              {LocType::Vertices, LocType::Cells}),
                                                          Op::multiply),
                                             b.lit(0.), {LocType::Edges, LocType::Cells}))),
                                     {LocType::Cells, LocType::Edges})))))),
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
          b.stage(LocType::Edges, b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
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
          b.stage(LocType::Cells, b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
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
                                  b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                                             b.stmt(b.assignExpr(b.at(f_v_out), b.at(f_v_in))))))));
  auto result = UnstructuredDimensionChecker::checkStageLocTypeConsistency(*stencil->getIIR(),
                                                                           stencil->getMetaData());
  EXPECT_EQ(result, UnstructuredDimensionChecker::ConsistencyResult(true, dawn::SourceLocation()));
}
TEST(UnstructuredDimensionCheckerTest, VarAccessType) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;

  auto varB = b.localvar("varX", dawn::BuiltinTypeID::Double, {b.lit(1.0)},
                         iir::LocalVariableType::OnEdges);

  auto stencil = b.build(
      "pass",
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(LocType::Edges,
                  b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                             b.declareVar(varB), b.stmt(b.assignExpr(b.at(varB), b.lit(2.0))))))));
  auto result = UnstructuredDimensionChecker::checkStageLocTypeConsistency(*stencil->getIIR(),
                                                                           stencil->getMetaData());
  EXPECT_EQ(result, UnstructuredDimensionChecker::ConsistencyResult(true, dawn::SourceLocation()));
}
TEST(UnstructuredDimensionCheckerTest, VerticalIndirectionFail) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto in = b.field("in", LocType::Cells);
  auto out = b.field("out", LocType::Cells);
  auto kidx = b.field("kidx", LocType::Edges);

  EXPECT_DEATH(
      auto stencil = b.build(
          "fail", b.stencil(b.multistage(
                      LoopOrderKind::Parallel,
                      b.stage(b.doMethod(
                          dawn::ast::Interval::Start, dawn::ast::Interval::End,
                          b.stmt(b.assignExpr(b.at(out), b.at(in, AccessType::r,
                                                              ast::Offsets{ast::unstructured, false,
                                                                           1, "kidx"})))))))),
      ".*Dimensions consistency check failed.*");
}
TEST(UnstructuredDimensionCheckerTest, VerticalIndirectionPass1) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto edge_field = b.field("edge_field", LocType::Edges);
  auto cell_field = b.field("cell_field", LocType::Cells);
  auto kidx_edge = b.field("kidx_edge", LocType::Edges);

  auto stencil = b.build(
      "pass",
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(b.doMethod(
              dawn::ast::Interval::Start, dawn::ast::Interval::End,
              b.stmt(b.assignExpr(b.at(cell_field),
                                  b.reduceOverNeighborExpr(
                                      Op::plus,
                                      b.at(edge_field, AccessType::r,
                                           ast::Offsets{ast::unstructured, false, 1, "kidx_edge"}),
                                      b.lit(0.), {LocType::Cells, LocType::Edges}))))))));

  auto result = UnstructuredDimensionChecker::checkDimensionsConsistency(*stencil->getIIR(),
                                                                         stencil->getMetaData());
  EXPECT_EQ(result, UnstructuredDimensionChecker::ConsistencyResult(true, dawn::SourceLocation()));
}
TEST(UnstructuredDimensionCheckerTest, VerticalIndirectionPass2) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto edge_field = b.field("edge_field", LocType::Edges);
  auto cell_field = b.field("cell_field", LocType::Cells);
  auto kidx_edge = b.field("kidx_sparse", {LocType::Edges, LocType::Cells});

  auto stencil = b.build(
      "pass",
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                             b.stmt(b.assignExpr(
                                 b.at(cell_field),
                                 b.reduceOverNeighborExpr(
                                     Op::plus,
                                     b.at(edge_field, AccessType::r,
                                          ast::Offsets{ast::unstructured, false, 1, "kidx_sparse"}),
                                     b.lit(0.), {LocType::Cells, LocType::Edges}))))))));

  auto result = UnstructuredDimensionChecker::checkDimensionsConsistency(*stencil->getIIR(),
                                                                         stencil->getMetaData());
  EXPECT_EQ(result, UnstructuredDimensionChecker::ConsistencyResult(true, dawn::SourceLocation()));
}
TEST(UnstructuredDimensionCheckerTest, IfStmt) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto cond = b.field("cond", LocType::Cells);
  auto body = b.field("body", LocType::Edges);

  EXPECT_DEATH(
      auto stencil = b.build(
          "fail", b.stencil(b.multistage(
                      LoopOrderKind::Parallel,
                      b.stage(b.doMethod(
                          dawn::ast::Interval::Start, dawn::ast::Interval::End,
                          b.ifStmt(b.at(cond), b.stmt(b.assignExpr(b.at(body), b.lit(1.))))))))),
      ".*Dimensions consistency check failed.*");
}

TEST(UnstructuredDimensionCheckerTest, IfStmtNest) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto cond = b.field("cond", LocType::Cells);
  auto inner_cond = b.field("inner_cond", LocType::Edges);
  auto inner_body = b.field("inner_body", LocType::Edges);

  EXPECT_DEATH(
      auto stencil = b.build(
          "fail", b.stencil(b.multistage(
                      LoopOrderKind::Parallel,
                      b.stage(b.doMethod(
                          dawn::ast::Interval::Start, dawn::ast::Interval::End,
                          b.ifStmt(b.at(cond), b.ifStmt(b.at(inner_cond),
                                                        b.stmt(b.assignExpr(b.at(inner_body),
                                                                            b.lit(1.)))))))))),
      ".*Dimensions consistency check failed.*");
}

TEST(UnstructuredDimensionCheckerTest, IfStmtNestPass) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto cond = b.field("cond", LocType::Edges);
  auto inner_cond = b.field("inner_cond", LocType::Edges);
  auto inner_body = b.field("inner_body", LocType::Edges);

  auto stencil = b.build(
      "pass",
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                             b.ifStmt(b.at(cond), b.ifStmt(b.at(inner_cond),
                                                           b.stmt(b.assignExpr(b.at(inner_body),
                                                                               b.lit(1.))))))))));

  auto result = UnstructuredDimensionChecker::checkDimensionsConsistency(*stencil->getIIR(),
                                                                         stencil->getMetaData());
  EXPECT_EQ(result, UnstructuredDimensionChecker::ConsistencyResult(true, dawn::SourceLocation()));
}

TEST(UnstructuredDimensionCheckerTest, IfStmtNestPassVert) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto cond = b.vertical_field("cond");
  auto inner_cond = b.field("inner_cond", LocType::Edges);
  auto inner_body = b.field("inner_body", LocType::Edges);

  auto stencil = b.build(
      "pass",
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(LocType::Edges,
                  b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                             b.ifStmt(b.at(cond), b.ifStmt(b.at(inner_cond),
                                                           b.stmt(b.assignExpr(b.at(inner_body),
                                                                               b.lit(1.))))))))));
  {
    auto result = UnstructuredDimensionChecker::checkDimensionsConsistency(*stencil->getIIR(),
                                                                           stencil->getMetaData());
    EXPECT_EQ(result,
              UnstructuredDimensionChecker::ConsistencyResult(true, dawn::SourceLocation()));
  }

  {
    auto result = UnstructuredDimensionChecker::checkStageLocTypeConsistency(
        *stencil->getIIR(), stencil->getMetaData());
    EXPECT_EQ(result,
              UnstructuredDimensionChecker::ConsistencyResult(true, dawn::SourceLocation()));
  }
}

TEST(UnstructuredDimensionCheckerTest, IfStmtVarAccess) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_c = b.field("f_c", ast::LocationType::Cells);
  auto f_e = b.field("f_e", ast::LocationType::Edges);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(2.0)});

  /// field(cells) f_c;
  /// field(edges) f_e;
  /// double varA = 2.0;
  /// if(f_e > 0.0) {
  ///    if(f_c > 0.0) {
  ///       varA = 1.0;
  ///    }
  /// }

  EXPECT_DEATH(
      auto stencil = b.build(
          "generated",
          b.stencil(b.multistage(
              dawn::iir::LoopOrderKind::Forward,
              b.stage(b.doMethod(
                  dawn::ast::Interval::Start, dawn::ast::Interval::End, b.declareVar(varA),
                  b.ifStmt(b.binaryExpr(b.at(f_e), b.lit(0.0), Op::greater),
                           b.block(b.ifStmt(
                               b.binaryExpr(b.at(f_c), b.lit(0.0), Op::greater),
                               b.block(b.stmt(b.assignExpr(b.at(varA), b.lit(1.0)))))))))))),
      ".*Dimensions consistency check failed.*");
}

TEST(UnstructuredDimensionCheckerTest, IfStmtThenBlock) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto f_c = b.field("f_c", LocType::Cells);
  auto f_e_1 = b.field("f_e_1", LocType::Edges);
  auto f_e_2 = b.field("f_e_2", LocType::Edges);

  EXPECT_DEATH(
      auto stencil = b.build(
          "fail",
          b.stencil(b.multistage(
              LoopOrderKind::Parallel,
              b.stage(b.doMethod(
                  dawn::ast::Interval::Start, dawn::ast::Interval::End,
                  b.ifStmt(b.at(f_c), b.block(b.stmt(b.assignExpr(b.at(f_e_1), b.lit(1.0))),
                                              b.stmt(b.assignExpr(b.at(f_e_2), b.lit(2.0)))))))))),
      ".*Dimensions consistency check failed.*");
}

TEST(UnstructuredDimensionCheckerTest, IfStmtElseBlock) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto f_c = b.field("f_c", LocType::Cells);
  auto f_e_1 = b.field("f_e_1", LocType::Edges);
  auto f_e_2 = b.field("f_e_2", LocType::Edges);

  EXPECT_DEATH(
      auto stencil = b.build(
          "fail",
          b.stencil(b.multistage(
              LoopOrderKind::Parallel,
              b.stage(b.doMethod(
                  dawn::ast::Interval::Start, dawn::ast::Interval::End,
                  b.ifStmt(b.at(f_c), b.block(b.stmt(b.assignExpr(b.at(f_e_1), b.lit(0.0)))),
                           b.block(b.stmt(b.assignExpr(b.at(f_e_1), b.lit(1.0))),
                                   b.stmt(b.assignExpr(b.at(f_e_2), b.lit(2.0)))))))))),
      ".*Dimensions consistency check failed.*");
}

TEST(UnstructuredDimensionCheckerTest, ThenBlockInconsistent) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto f_vertical = b.vertical_field("f_vertical");
  auto f_c = b.field("f_c", LocType::Cells);
  auto f_e = b.field("f_e", LocType::Edges);

  EXPECT_DEATH(
      auto stencil = b.build(
          "fail", b.stencil(b.multistage(
                      LoopOrderKind::Parallel,
                      b.stage(b.doMethod(
                          dawn::ast::Interval::Start, dawn::ast::Interval::End,
                          b.ifStmt(b.at(f_vertical),
                                   b.block(b.stmt(b.assignExpr(b.at(f_c), b.lit(1.0))),
                                           b.stmt(b.assignExpr(b.at(f_e), b.lit(2.0)))))))))),
      ".*Dimensions consistency check failed.*");
}

TEST(UnstructuredDimensionCheckerTest, ThenBlockElseBlockInconsistent) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto f_vertical = b.vertical_field("f_vertical");
  auto f_c = b.field("f_c", LocType::Cells);
  auto f_e = b.field("f_e", LocType::Edges);

  EXPECT_DEATH(
      auto stencil = b.build(
          "fail",
          b.stencil(b.multistage(
              LoopOrderKind::Parallel,
              b.stage(b.doMethod(
                  dawn::ast::Interval::Start, dawn::ast::Interval::End,
                  b.ifStmt(b.at(f_vertical), b.block(b.stmt(b.assignExpr(b.at(f_c), b.lit(1.0)))),
                           b.block(b.stmt(b.assignExpr(b.at(f_e), b.lit(2.0)))))))))),
      ".*Dimensions consistency check failed.*");
}

TEST(UnstructuredDimensionCheckerTest, ThenBlockVertical) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto f_vertical = b.vertical_field("f_vertical");
  auto f_vertical_in = b.vertical_field("f_vertical_in");
  auto f_e = b.field("f_e", LocType::Edges);

  auto stencil = b.build(
      "pass", b.stencil(b.multistage(
                  LoopOrderKind::Parallel,
                  b.stage(b.doMethod(
                      dawn::ast::Interval::Start, dawn::ast::Interval::End,
                      b.ifStmt(b.at(f_vertical),
                               b.block(b.stmt(b.assignExpr(b.at(f_vertical_in), b.lit(1.0)))),
                               b.block(b.stmt(b.assignExpr(b.at(f_e), b.lit(2.0))))))))));
  {
    auto result = UnstructuredDimensionChecker::checkDimensionsConsistency(*stencil->getIIR(),
                                                                           stencil->getMetaData());
    EXPECT_EQ(result,
              UnstructuredDimensionChecker::ConsistencyResult(true, dawn::SourceLocation()));
  }
}

TEST(UnstructuredDimensionCheckerTest, ElseBlockVertical) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto f_vertical = b.vertical_field("f_vertical");
  auto f_vertical_in = b.vertical_field("f_vertical_in");
  auto f_e = b.field("f_e", LocType::Edges);

  auto stencil = b.build(
      "pass",
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(b.doMethod(
              dawn::ast::Interval::Start, dawn::ast::Interval::End,
              b.ifStmt(b.at(f_vertical), b.block(b.stmt(b.assignExpr(b.at(f_e), b.lit(1.0)))),
                       b.block(b.stmt(b.assignExpr(b.at(f_vertical_in), b.lit(2.0))))))))));
  {
    auto result = UnstructuredDimensionChecker::checkDimensionsConsistency(*stencil->getIIR(),
                                                                           stencil->getMetaData());
    EXPECT_EQ(result,
              UnstructuredDimensionChecker::ConsistencyResult(true, dawn::SourceLocation()));
  }
}

TEST(UnstructuredDimensionCheckerTest, DimsWithCenterCorrect) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto cin_f = b.field("cin_field", LocType::Cells);
  auto cout_f = b.field("cout_field", LocType::Cells);
  auto sparse_f = b.field("sparse", {LocType::Cells, LocType::Edges, LocType::Cells},
                          /*maskK*/ true, /*include_center*/ true);

  auto stencil = b.build(
      "pass",
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(LocType::Cells,
                  b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                             b.loopStmtChain(b.stmt(b.assignExpr(b.at(sparse_f), b.lit(2.))),
                                             {LocType::Cells, LocType::Edges, LocType::Cells},
                                             /*include center*/ true),
                             b.stmt(b.assignExpr(
                                 b.at(cout_f),
                                 b.reduceOverNeighborExpr(
                                     Op::plus,
                                     b.binaryExpr(b.at(cin_f, HOffsetType::withOffset, 0),
                                                  b.at(sparse_f), Op::multiply),
                                     b.lit(0.), {LocType::Cells, LocType::Edges, LocType::Cells},
                                     /*include center*/ true))))))));
  {
    auto result = UnstructuredDimensionChecker::checkDimensionsConsistency(*stencil->getIIR(),
                                                                           stencil->getMetaData());
    EXPECT_EQ(result,
              UnstructuredDimensionChecker::ConsistencyResult(true, dawn::SourceLocation()));
  }
}

TEST(UnstructuredDimensionCheckerTest, DimsWithCenterFailFill) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto cin_f = b.field("cin_field", LocType::Cells);
  auto cout_f = b.field("cout_field", LocType::Cells);
  auto sparse_f = b.field("sparse", {LocType::Cells, LocType::Edges, LocType::Cells},
                          /*maskK*/ true, /*include_center*/ false);

  EXPECT_DEATH(
      b.build(
          "fail",
          b.stencil(b.multistage(
              LoopOrderKind::Parallel,
              b.stage(LocType::Cells,
                      b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                                 b.loopStmtChain(b.stmt(b.assignExpr(b.at(sparse_f), b.lit(2.))),
                                                 {LocType::Cells, LocType::Edges, LocType::Cells},
                                                 /*include center*/ true)))))),
      ".*Dimensions consistency check failed.*");
}

TEST(UnstructuredDimensionCheckerTest, DimsWithCenterFailReduce) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto cin_f = b.field("cin_field", LocType::Cells);
  auto cout_f = b.field("cout_field", LocType::Cells);
  auto sparse_f = b.field("sparse", {LocType::Cells, LocType::Edges, LocType::Cells},
                          /*maskK*/ true, /*include_center*/ false);

  EXPECT_DEATH(
      b.build("fail",
              b.stencil(b.multistage(
                  LoopOrderKind::Parallel,
                  b.stage(LocType::Cells,
                          b.doMethod(
                              dawn::ast::Interval::Start, dawn::ast::Interval::End,
                              b.stmt(b.assignExpr(
                                  b.at(cout_f),
                                  b.reduceOverNeighborExpr(
                                      Op::plus,
                                      b.binaryExpr(b.at(cin_f, HOffsetType::withOffset, 0),
                                                   b.at(sparse_f), Op::multiply),
                                      b.lit(0.), {LocType::Cells, LocType::Edges, LocType::Cells},
                                      /*include center*/ true)))))))),
      ".*Dimensions consistency check failed.*");
}

} // namespace