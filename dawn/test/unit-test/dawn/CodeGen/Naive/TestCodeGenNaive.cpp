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
#include <filesystem>

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

TEST(CodeGenNaiveTest, NonOverlappingInterval) {
  using namespace dawn::iir;
  using SInterval = dawn::sir::Interval;

  CartesianIIRBuilder b;
  auto in = b.field("in", FieldType::ijk);
  auto out = b.field("out", FieldType::ijk);
  auto dx = b.localvar("dx", dawn::BuiltinTypeID::Double);

  auto stencil_inst = b.build("generated",
    b.stencil(
      b.multistage(LoopOrderKind::Parallel,
        b.stage(
          b.doMethod(SInterval(SInterval::Start, 10), b.declareVar(dx),
            b.block(
              b.stmt(
                b.assignExpr(b.at(out),
                  b.binaryExpr(
                    b.binaryExpr(b.lit(-4),
                      b.binaryExpr(b.at(in),
                        b.binaryExpr(b.at(in, {1, 0, 0}),
                          b.binaryExpr(b.at(in, {-1, 0, 0}),
                            b.binaryExpr(b.at(in, {0, -1, 0}), b.at(in, {0, 1, 0}))
                    ) ) ), Op::multiply),
                    b.binaryExpr(b.at(dx), b.at(dx), Op::multiply), Op::divide)
            ) ) ) ) )
         , b.stage(b.doMethod(SInterval(15, SInterval::End),
            b.block(
              b.stmt(
                b.assignExpr(b.at(out), b.lit(10))
  ) ) ) ) ) ) );

  std::ostringstream oss;
  dump(oss, stencil_inst);
  std::string gen = oss.str();

  std::string ref = read("reference/nonoverlapping_stencil.cpp");
  ASSERT_EQ(gen, ref) << "Generated code does not match reference code";
}

TEST(CodeGenNaiveTest, LaplacianStencil) {
  using namespace dawn::iir;
  using SInterval = dawn::sir::Interval;

  CartesianIIRBuilder b;
  auto in = b.field("in", FieldType::ijk);
  auto out = b.field("out", FieldType::ijk);
  auto dx = b.localvar("dx", dawn::BuiltinTypeID::Double);

  auto stencil_inst = b.build("generated",
    b.stencil(
      b.multistage(LoopOrderKind::Parallel,
        b.stage(
          b.doMethod(SInterval::Start, SInterval::End, b.declareVar(dx),
            b.block(
              b.stmt(
                b.assignExpr(b.at(out),
                  b.binaryExpr(
                    b.binaryExpr(b.lit(-4),
                      b.binaryExpr(b.at(in),
                        b.binaryExpr(b.at(in, {1, 0, 0}),
                          b.binaryExpr(b.at(in, {-1, 0, 0}),
                            b.binaryExpr(b.at(in, {0, -1, 0}), b.at(in, {0, 1, 0}))
                    ) ) ), Op::multiply),
                    b.binaryExpr(b.at(dx), b.at(dx), Op::multiply), Op::divide)
            ) ) ) ) )
          ) ) );

  std::ofstream ofs("test/unit-test/dawn/CodeGen/Naive/generated/laplacian_stencil.cpp");
  dump(ofs, stencil_inst);
}

} // anonymous namespace
