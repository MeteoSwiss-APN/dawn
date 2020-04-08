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
#include "dawn/Validator/WeightChecker.h"

#include <gtest/gtest.h>
#include <memory>

using namespace dawn;

namespace {
TEST(WeightCheckerTest, Reduce_0) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto edge_field = b.field("edge_field", LocType::Edges);
  auto cell_field = b.field("cell_field", LocType::Cells);

  auto stencil = b.build(
      "pass",
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(b.doMethod(
              dawn::sir::Interval::Start, dawn::sir::Interval::End,
              b.stmt(b.assignExpr(
                  b.at(cell_field),
                  b.reduceOverNeighborExpr(Op::plus, b.at(edge_field, HOffsetType::withOffset, 0),
                                           b.lit(0.), {LocType::Cells, LocType::Edges},
                                           std::vector<double>{1., 2., 3.}))))))));
  auto result = WeightChecker::CheckWeights(*stencil->getIIR(), stencil->getMetaData());
  EXPECT_EQ(result, WeightChecker::ConsistencyResult(true, dawn::SourceLocation()));
}

TEST(WeightCheckerTest, Reduce_1) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto edge_field = b.field("edge_field", LocType::Edges);
  auto cell_field = b.field("cell_field", LocType::Cells);
  auto distToE1 = b.field("dist_to_e1", LocType::Cells);
  auto distToE2 = b.field("dist_to_e2", LocType::Cells);
  auto distToE3 = b.field("dist_to_e3", LocType::Cells);

  auto stencil = b.build(
      "pass",
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(b.doMethod(
              dawn::sir::Interval::Start, dawn::sir::Interval::End,
              b.stmt(b.assignExpr(b.at(cell_field),
                                  b.reduceOverNeighborExpr(
                                      Op::plus, b.at(edge_field, HOffsetType::withOffset, 0),
                                      b.lit(0.), {LocType::Cells, LocType::Edges},
                                      std::vector<std::shared_ptr<Expr>>{
                                          b.at(distToE1), b.at(distToE2), b.at(distToE3)}))))))));

  auto result = WeightChecker::CheckWeights(*stencil->getIIR(), stencil->getMetaData());
  EXPECT_EQ(result, WeightChecker::ConsistencyResult(true, dawn::SourceLocation()));
}

TEST(WeightCheckerTest, Reduce_2) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto edge_field = b.field("edge_field", LocType::Edges);
  auto cell_field = b.field("cell_field", LocType::Cells);
  auto distToE = b.field("dist_to_e", {LocType::Cells, LocType::Edges});

  EXPECT_DEATH(
      b.build(
          "fail",
          b.stencil(b.multistage(
              LoopOrderKind::Parallel,
              b.stage(b.doMethod(
                  dawn::sir::Interval::Start, dawn::sir::Interval::End,
                  b.stmt(b.assignExpr(b.at(cell_field),
                                      b.reduceOverNeighborExpr(
                                          Op::plus, b.at(edge_field, HOffsetType::withOffset, 0),
                                          b.lit(0.), {LocType::Cells, LocType::Edges},
                                          std::vector<std::shared_ptr<Expr>>{
                                              b.at(distToE), b.at(distToE), b.at(distToE)})))))))),
      ".*Found invalid weights.*");
}

TEST(WeightCheckerTest, NestedReduce_0) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto cell_f = b.field("cell_field", LocType::Cells);
  auto edge_f = b.field("edge_field", LocType::Edges);
  auto vertex_f = b.field("vertex_field", LocType::Vertices);

  // a nested reduction v->e->c
  auto stencil = b.build(
      "pass",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Parallel,
          b.stage(LocType::Cells,
                  b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                             b.stmt(b.assignExpr(
                                 b.at(cell_f),
                                 b.reduceOverNeighborExpr(
                                     Op::plus,
                                     b.reduceOverNeighborExpr(Op::plus, b.at(vertex_f), b.lit(0.),
                                                              {LocType::Edges, LocType::Vertices},
                                                              std::vector<std::shared_ptr<Expr>>{
                                                                  b.at(edge_f), b.at(edge_f)}),
                                     b.lit(0.), {LocType::Cells, LocType::Edges},
                                     std::vector<std::shared_ptr<Expr>>{b.at(cell_f),
                                                                        b.at(cell_f)}))))))));

  auto result = WeightChecker::CheckWeights(*stencil->getIIR(), stencil->getMetaData());
  EXPECT_EQ(result, WeightChecker::ConsistencyResult(true, dawn::SourceLocation()));
}

TEST(WeightCheckerTest, NestedReduce_1) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto cell_f = b.field("cell_field", LocType::Cells);
  auto edge_f = b.field("edge_field", LocType::Edges);
  auto vertex_f = b.field("vertex_field", LocType::Vertices);
  auto distToE = b.field("dist_to_e", {LocType::Cells, LocType::Edges});

  // a nested reduction v->e->c
  EXPECT_DEATH(
      b.build("fail",
              b.stencil(b.multistage(
                  dawn::iir::LoopOrderKind::Parallel,
                  b.stage(LocType::Cells,
                          b.doMethod(
                              dawn::sir::Interval::Start, dawn::sir::Interval::End,
                              b.stmt(b.assignExpr(
                                  b.at(cell_f),
                                  b.reduceOverNeighborExpr(
                                      Op::plus,
                                      b.reduceOverNeighborExpr(Op::plus, b.at(vertex_f), b.lit(0.),
                                                               {LocType::Edges, LocType::Vertices},
                                                               std::vector<std::shared_ptr<Expr>>{
                                                                   b.at(edge_f), b.at(edge_f)}),
                                      b.lit(0.), {LocType::Cells, LocType::Edges},
                                      std::vector<std::shared_ptr<Expr>>{b.at(distToE),
                                                                         b.at(distToE)})))))))),
      ".*Found invalid weights.*");
}

} // namespace