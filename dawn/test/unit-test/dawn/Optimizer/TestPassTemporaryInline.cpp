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
#include "dawn/CodeGen/Driver.h"
#include "dawn/Optimizer/PassTemporaryInlining.h"
#include "dawn/Support/Logger.h"
#include "dawn/Unittest/ASTConstructionAliases.h"
#include "dawn/Unittest/IIRBuilder.h"
#include "dawn/Unittest/UnittestUtils.h"

#include <gtest/gtest.h>
#include <sstream>

using namespace dawn;
using namespace astgen;

namespace {

TEST(TestTemporaryInline, test_temporary_inline_01) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto outF = b.field("outF", ast::LocationType::Vertices);
  auto inF = b.field("inF", ast::LocationType::Cells);
  auto tempF = b.tmpField("tempF", ast::LocationType::Edges);

  auto stencil = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(
              LocType::Edges,
              b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                         b.stmt(b.assignExpr(b.at(tempF), b.reduceOverNeighborExpr(
                                                              Op::plus, b.at(inF), b.lit(0.),
                                                              {LocType::Edges, LocType::Cells}))))),
          b.stage(LocType::Vertices,
                  b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                             b.stmt(b.assignExpr(b.at(outF),
                                                 b.reduceOverNeighborExpr(
                                                     Op::plus, b.at(tempF), b.lit(0.),
                                                     {LocType::Vertices, LocType::Edges}))))))));

  PassTemporaryInlining passTemporaryInline;
  passTemporaryInline.run(stencil);

  ASSERT_EQ(getFirstDoMethod(stencil).getAST().getStatements().size(), 1);

  auto firstStatement = getNthStmt(getFirstDoMethod(stencil), 0);

  ASSERT_TRUE(firstStatement->equals(
      expr(assign(field("outF"), red(red(field("inF"), {LocType::Edges, LocType::Cells}),
                                     {LocType::Vertices, LocType::Edges})))
          .get(),
      /*compareData = */ false));
}

TEST(TestTemporaryInline, test_temporary_inline_02) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto outF = b.field("outF", ast::LocationType::Vertices);
  auto inF = b.field("inF", ast::LocationType::Cells);
  auto tempF = b.tmpField("tempF", ast::LocationType::Edges);

  auto stencil = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(b.doMethod(
              dawn::ast::Interval::Start, dawn::ast::Interval::End,
              b.stmt(b.assignExpr(b.at(tempF),
                                  b.reduceOverNeighborExpr(Op::plus, b.at(inF), b.lit(0.),
                                                           {LocType::Edges, LocType::Cells}))),
              b.stmt(b.assignExpr(b.at(tempF), b.lit(1.))),
              b.stmt(b.assignExpr(
                  b.at(outF), b.reduceOverNeighborExpr(Op::plus, b.at(tempF), b.lit(0.),
                                                       {LocType::Vertices, LocType::Edges}))))))));

  auto stencilBefore = stencil->clone();
  PassTemporaryInlining passTemporaryInline;
  passTemporaryInline.run(stencil);

  // lets check that stmts were left alone by the pass
  ASSERT_EQ(stencil->getStencils().size(), stencilBefore->getStencils().size());

  for(int stenIdx = 0; stenIdx < stencil->getStencils().size(); stenIdx++) {
    auto stmtsBefore = iterateIIROverStmt(*stencilBefore->getStencils()[stenIdx]);
    auto stmtsAfter = iterateIIROverStmt(*stencil->getStencils()[stenIdx]);

    ASSERT_EQ(stmtsBefore.size(), stmtsAfter.size());

    for(auto stmtIdx = 0; stmtIdx < stmtsBefore.size(); stmtIdx++) {
      ASSERT_TRUE(stmtsBefore[stmtIdx]->equals(stmtsAfter[stmtIdx].get()));
    }
  }
}

TEST(TestTemporaryInline, test_temporary_inline_03) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto outF = b.field("outF", ast::LocationType::Vertices);
  auto inFLft = b.field("inFLft", ast::LocationType::Edges);
  auto inFRgt = b.field("inFRgt", ast::LocationType::Edges);
  auto tempF = b.tmpField("tempF", ast::LocationType::Edges);

  auto stencil = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(
              LocType::Edges,
              b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                         b.stmt(b.assignExpr(b.at(tempF),
                                             b.binaryExpr(b.at(inFLft), b.at(inFRgt), Op::plus))))),
          b.stage(LocType::Vertices,
                  b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                             b.stmt(b.assignExpr(b.at(outF),
                                                 b.reduceOverNeighborExpr(
                                                     Op::plus, b.at(tempF), b.lit(0.),
                                                     {LocType::Vertices, LocType::Edges}))))))));

  PassTemporaryInlining passTemporaryInline;
  passTemporaryInline.run(stencil);

  ASSERT_EQ(getFirstDoMethod(stencil).getAST().getStatements().size(), 1);

  auto firstStatement = getNthStmt(getFirstDoMethod(stencil), 0);

  ASSERT_TRUE(firstStatement->equals(
      expr(assign(field("outF"), red(binop(field("inFLft"), "+", field("inFRgt")),
                                     {LocType::Vertices, LocType::Edges})))
          .get(),
      /*compareData = */ false));
}

} // namespace