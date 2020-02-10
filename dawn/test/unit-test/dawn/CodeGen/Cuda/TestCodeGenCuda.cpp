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
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/DiagnosticsEngine.h"
#include "dawn/Support/FileUtil.h"
#include "dawn/Unittest/CompilerUtil.h"
#include "dawn/Unittest/IIRBuilder.h"
#include "dawn/Unittest/UnittestLogger.h"

#include <gtest/gtest.h>

namespace {

TEST(CodeGenCudaTest, GlobalIndexStencil) {
  using namespace dawn::iir;

  CartesianIIRBuilder b;
  auto in_f = b.field("in_field", FieldType::ijk);
  auto out_f = b.field("out_field", FieldType::ijk);

  auto stencil_instantiation =
      b.build("generated",
              b.stencil(b.multistage(
                  LoopOrderKind::Parallel,
                  b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                     b.block(b.stmt(b.assignExpr(b.at(out_f), b.at(in_f)))))),
                  b.stage(1, {0, 2},
                          b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                     b.block(b.stmt(b.assignExpr(b.at(out_f), b.lit(10)))))))));

  std::ostringstream oss;
  dawn::CompilerUtil::dumpCuda(oss, stencil_instantiation);
  std::string gen = oss.str();

  std::string ref = dawn::readFile("reference/global_indexing.cpp");
  ASSERT_EQ(gen, ref) << "Generated code does not match reference code";
}

} // anonymous namespace
