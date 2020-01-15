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

#include "dawn/CodeGen/CodeGen.h"
#include "dawn/CodeGen/CXXNaive/CXXNaiveCodeGen.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/DiagnosticsEngine.h"
#include "dawn/Unittest/IIRBuilder.h"
#include "dawn/Unittest/UnittestLogger.h"

#include <gtest/gtest.h>

#include <cstring>
#include <fstream>

namespace {

void dump(std::ostream& os, dawn::codegen::stencilInstantiationContext& ctx) {
  using CG = dawn::codegen::cxxnaive::CXXNaiveCodeGen;
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

std::string read(const std::string& file) {
  std::ifstream is(file);
  std::string str((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
  return str;
}

TEST(CodeGenNaiveTest, GlobalIndexStencil) {
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

  std::ostringstream oss;
  dump(oss, stencil_instantiation);
  std::string gen = oss.str();

  std::string ref = read("test/unit-test/dawn/CodeGen/Naive/generated/global_indexing.cpp");
  ASSERT_EQ(gen, ref) << "Generated code does not match reference code";
}

} // anonymous namespace
