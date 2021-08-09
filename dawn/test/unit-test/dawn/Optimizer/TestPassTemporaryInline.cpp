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

  std::ofstream of("generated_InlineTest.hpp");
  DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
  auto tu = dawn::codegen::run(stencil, dawn::codegen::Backend::CXXNaiveIco);
  of << dawn::codegen::generate(tu) << std::endl;
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

  PassTemporaryInlining passTemporaryInline;
  passTemporaryInline.run(stencil);

  std::ofstream of("generated/generated_InlineTest.hpp");
  DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
  auto tu = dawn::codegen::run(stencil, dawn::codegen::Backend::CXXNaiveIco);
  of << dawn::codegen::generate(tu) << std::endl;
}

} // namespace