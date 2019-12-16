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

#include "dawn-c/Compiler.h"
#include "dawn-c/TranslationUnit.h"
#include "dawn/CodeGen/CXXNaive-ico/CXXNaiveCodeGen.h"
#include "dawn/CodeGen/CXXNaive/CXXNaiveCodeGen.h"
#include "dawn/CodeGen/CodeGen.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "dawn/Support/DiagnosticsEngine.h"
#include "dawn/Unittest/IIRBuilder.h"

#include <gtest/gtest.h>

#include <cstring>
#include <fstream>

namespace {

static void freeCharArray(char** array, int size) {
  for(int i = 0; i < size; ++i)
    std::free(array[i]);
  std::free(array);
}

TEST(CompilerTest, CompileEmptySIR) {
  std::string sir;
  dawnTranslationUnit_t* TU = dawnCompile(sir.data(), sir.size(), nullptr);

  EXPECT_EQ(dawnTranslationUnitGetStencil(TU, "invalid"), nullptr);

  char** ppDefines;
  int size;
  dawnTranslationUnitGetPPDefines(TU, &ppDefines, &size);
  EXPECT_NE(size, 0);
  EXPECT_NE(ppDefines, nullptr);

  freeCharArray(ppDefines, size);
  dawnTranslationUnitDestroy(TU);
}
template <typename CG>
void dump(std::ostream& os, dawn::codegen::stencilInstantiationContext& ctx) {
  dawn::DiagnosticsEngine diagnostics;
  CG generator(ctx, diagnostics, 0);
  auto tu = generator.generateCode();

  std::ostringstream ss;
  for(auto const& macroDefine : tu->getPPDefines())
    ss << macroDefine << "\n";

  ss << tu->getGlobals();
  for(auto const& s : tu->getStencils())
    ss << s.second;
  os << ss.str();
}
TEST(CompilerTest, CompileCopyStencil) {
  using namespace dawn::iir;

  CartesianIIRBuilder b;
  auto in_f = b.field("in_field", FieldType::ijk);
  auto out_f = b.field("out_field", FieldType::ijk);

  auto stencil_instantiation =
      b.build("generated",
              b.stencil(b.multistage(
                  LoopOrderKind::Parallel,
                  b.stage(b.vregion(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                    b.block(b.stmt(b.assignExpr(b.at(out_f), b.at(in_f)))))))));
  std::ofstream of("/dev/null");
  dump<dawn::codegen::cxxnaive::CXXNaiveCodeGen>(of, stencil_instantiation);
}

TEST(CompilerTest, CompileGlobalIndexStencil) {
  using namespace dawn::iir;

  CartesianIIRBuilder b;
  auto in_f = b.field("in_field", FieldType::ijk);
  auto out_f = b.field("out_field", FieldType::ijk);

  auto stencil_instantiation = b.build(
      "generated", b.stencil(b.multistage(
                       LoopOrderKind::Parallel,
                       b.stage(b.vregion(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                         b.block(b.stmt(b.assignExpr(b.at(out_f), b.at(in_f)))))),
                       b.stage(1, {0, 2},
                               b.vregion(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                         b.block(b.stmt(b.assignExpr(b.at(out_f), b.lit(10)))))))));
  std::ofstream of("prototype/generated/global_indexing.cpp");
  dump<dawn::codegen::cxxnaive::CXXNaiveCodeGen>(of, stencil_instantiation);
}

TEST(CompilerTest, DISABLED_CodeGenSumEdgeToCells) {
  using namespace dawn::iir;
  using LocType = dawn::ast::Expr::LocationType;

  UnstructuredIIRBuilder b;
  auto in_f = b.field("in_field", LocType::Edges);
  auto out_f = b.field("out_field", LocType::Cells);
  auto cnt = b.localvar("cnt", dawn::BuiltinTypeID::Integer);

  auto stencil_instantiation = b.build(
      "generated",
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(LocType::Edges, b.vregion(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                            b.stmt(b.assignExpr(b.at(in_f), b.lit(10))))),
          b.stage(b.vregion(
              dawn::sir::Interval::Start, dawn::sir::Interval::End,
              b.stmt(b.assignExpr(b.at(out_f), b.reduceOverNeighborExpr(
                                                   Op::plus, b.at(in_f, HOffsetType::withOffset, 0),
                                                   b.lit(0.), LocType::Edges))))))));

  std::ofstream of("prototype/generated_copyEdgeToCell.hpp");
  DAWN_ASSERT_MSG(of, "file could not be opened. Binary must be called from dawn/dawn");
  dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
  of.close();
}

TEST(CompilerTest, DISABLED_SumVertical) {
  using namespace dawn::iir;
  using LocType = dawn::ast::Expr::LocationType;

  UnstructuredIIRBuilder b;
  auto in_f = b.field("in_field", LocType::Cells);
  auto out_f = b.field("out_field", LocType::Cells);

  auto stencil_instantiation = b.build(
      "generated",
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(LocType::Cells,
                  b.vregion(dawn::sir::Interval::Start, dawn::sir::Interval::End, 1, -1,
                            b.stmt(b.assignExpr(b.at(out_f),
                                                b.binaryExpr(b.at(in_f, HOffsetType::noOffset, +1),
                                                             b.at(in_f, HOffsetType::noOffset, -1),
                                                             Op::plus))))))));

  std::ofstream of("prototype/generated_verticalSum.hpp");
  DAWN_ASSERT_MSG(of, "file could not be opened. Binary must be called from dawn/dawn");
  dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
  of.close();
}

TEST(CompilerTest, DISABLED_CodeGenDiffusion) {
  using namespace dawn::iir;
  using LocType = dawn::ast::Expr::LocationType;

  UnstructuredIIRBuilder b;
  auto in_f = b.field("in_field", LocType::Cells);
  auto out_f = b.field("out_field", LocType::Cells);
  auto cnt = b.localvar("cnt", dawn::BuiltinTypeID::Integer);

  auto stencil_instantiation = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Parallel,
          b.stage(b.vregion(
              dawn::sir::Interval::Start, dawn::sir::Interval::End, b.declareVar(cnt),
              b.stmt(b.assignExpr(b.at(cnt),
                                  b.reduceOverNeighborExpr(Op::plus, b.lit(1), b.lit(0),
                                                           dawn::ast::Expr::LocationType::Cells))),
              b.stmt(b.assignExpr(
                  b.at(out_f),
                  b.reduceOverNeighborExpr(Op::plus, b.at(in_f, HOffsetType::withOffset, 0),
                                           b.binaryExpr(b.unaryExpr(b.at(cnt), Op::minus),
                                                        b.at(in_f, HOffsetType::withOffset, 0),
                                                        Op::multiply),
                                           dawn::ast::Expr::LocationType::Cells))),
              b.stmt(b.assignExpr(b.at(out_f),
                                  b.binaryExpr(b.at(in_f),
                                               b.binaryExpr(b.lit(0.1), b.at(out_f), Op::multiply),
                                               Op::plus))))))));

  std::ofstream of("prototype/generated_Diffusion.hpp");
  DAWN_ASSERT_MSG(of, "file could not be opened. Binary must be called from dawn/dawn");
  dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
  of.close();
}

} // anonymous namespace
