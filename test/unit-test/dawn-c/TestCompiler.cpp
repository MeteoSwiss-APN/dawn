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
#include "dawn/IIR/IIRBuilder.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "dawn/Support/DiagnosticsEngine.h"
#include "dawn/Unittest/ASTSimplifier.h"
#include <cstring>
#include <gtest/gtest.h>

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

TEST(CompilerTest, CompileCopyStencil) {
  using namespace dawn::astgen;

  // Build copy stencil
  //
  //  copy {
  //    storage in, out;
  //
  //    vertical_region(start, end) {
  //      out = in;
  //    }
  //  }
  //
  auto sir = std::make_shared<dawn::SIR>();
  auto stencil = std::make_shared<dawn::sir::Stencil>();
  stencil->Name = "copy";
  stencil->Fields.emplace_back(std::make_shared<dawn::sir::Field>("in"));
  stencil->Fields.emplace_back(std::make_shared<dawn::sir::Field>("out"));

  auto ast = std::make_shared<dawn::sir::AST>(block(assign(field("out"), field("in"))));
  auto vr = std::make_shared<dawn::sir::VerticalRegion>(
      ast,
      std::make_shared<dawn::sir::Interval>(dawn::sir::Interval::Start, dawn::sir::Interval::End),
      dawn::sir::VerticalRegion::LK_Forward);
  stencil->StencilDescAst = std::make_shared<dawn::sir::AST>(block(verticalRegion(vr)));
  sir->Stencils.emplace_back(stencil);

  std::string sirStr =
      dawn::SIRSerializer::serializeToString(sir.get(), dawn::SIRSerializer::SK_Byte);
  dawnTranslationUnit_t* TU = dawnCompile(sirStr.data(), sirStr.size(), nullptr);

  char* copyCode = dawnTranslationUnitGetStencil(TU, "copy");
  EXPECT_NE(copyCode, nullptr);

  char** ppDefines;
  int size;
  dawnTranslationUnitGetPPDefines(TU, &ppDefines, &size);
  EXPECT_NE(size, 0);
  EXPECT_NE(ppDefines, nullptr);

  freeCharArray(ppDefines, size);
  std::free(copyCode);
  dawnTranslationUnitDestroy(TU);
}

TEST(CompilerTest, TestCodeGen) {
  using namespace dawn::iir;

  IIRBuilder b;
  auto in_f = b.make_field("in_field", field_type::ijk);
  auto out_f = b.make_field("out_field", field_type::ijk);
  auto var = b.make_localvar("my_var");
  auto var2 = b.make_localvar("my_var2");

  auto stencil_instantiation = b.build(
      "generated",
      b.make_stencil(b.make_multistage(
          dawn::iir::LoopOrderKind::LK_Parallel,
          b.make_stage(b.make_do(
              dawn::sir::Interval::Start, dawn::sir::Interval::End, b.declare_var(var),
              b.make_stmt(b.make_assign_expr(
                  b.at(out_f, access_type::rw),
                  b.make_binary_expr(b.make_lit(-3.), b.make_unary_expr(b.at(in_f), op::minus),
                                     op::multiply))),
              b.make_stmt(b.make_assign_expr(
                  b.at(out_f, access_type::rw),
                  b.make_reduce_over_neighbor_expr(
                      op::plus, b.make_unary_expr(b.at(in_f), op::minus), b.at(out_f)))),
              b.make_stmt(
                  b.make_assign_expr(b.at(out_f, access_type::rw), b.make_lit(0.1), op::multiply)),
              b.make_stmt(b.make_assign_expr(b.at(out_f, access_type::rw), b.at(in_f, {0, 0, 1}),
                                             op::plus)))),
          b.make_stage(
              b.make_do(dawn::sir::Interval::Start, dawn::sir::Interval::End, b.declare_var(var2),
                        b.make_stmt(b.make_assign_expr(
                            b.at(out_f, access_type::rw),
                            b.make_binary_expr(b.make_lit(-3.),
                                               b.make_unary_expr(b.at(out_f, {0, 1, 0}), op::minus),
                                               op::multiply))),
                        b.make_stmt(b.make_assign_expr(b.at(out_f, access_type::rw),
                                                       b.make_lit(0.1), op::multiply)),
                        b.make_stmt(b.make_assign_expr(b.at(out_f, access_type::rw),
                                                       b.at(in_f, {0, 0, 1}), op::plus)))))));

  dawn::DiagnosticsEngine diagnostics;

  // TODO this should be moved into the IIR builder -> the IIR builder cannot be in IIR
  auto optimizer = dawn::make_unique<dawn::OptimizerContext>(
      diagnostics, dawn::OptimizerContext::OptimizerContextOptions{}, nullptr);
  optimizer->restoreIIR("<restored>", stencil_instantiation);
  stencil_instantiation = optimizer->getStencilInstantiationMap()["<restored>"];

  dawn::codegen::stencilInstantiationContext map;
  map["test"] = std::move(stencil_instantiation);

  // dawn::codegen::cxxnaive::CXXNaiveCodeGen generator(map, diagnostics, 0);
  dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen generator(map, diagnostics, 0);
  auto tu = generator.generateCode();

  std::ostringstream ss;
  for(auto const& macroDefine : tu->getPPDefines())
    ss << macroDefine << "\n";

  ss << tu->getGlobals();
  for(auto const& s : tu->getStencils())
    ss << s.second;
  std::clog << ss.str();
}

} // anonymous namespace
