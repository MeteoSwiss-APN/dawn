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

#include "dawn/CodeGen/CXXNaive/CXXNaiveCodeGen.h"
#include "dawn/CodeGen/CodeGen.h"
#include "dawn/CodeGen/Cuda/CudaCodeGen.h"
#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/Optimizer/PassSetCaches.h"
#include "dawn/Unittest/IIRBuilder.h"

#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <unistd.h>

using namespace dawn;

namespace {

template <typename CG>
void dump_cuda(std::ostream& os, dawn::codegen::stencilInstantiationContext& ctx) {
  dawn::DiagnosticsEngine diagnostics;
  CG generator(ctx, diagnostics, 0, 32, 64, {128, 128, 128});
  auto tu = generator.generateCode();

  std::ostringstream ss;
  for(auto const& macroDefine : tu->getPPDefines())
    ss << macroDefine << "\n";

  ss << tu->getGlobals();
  for(auto const& s : tu->getStencils())
    ss << s.second;
  os << ss.str();
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

// this is a sanity check, no global iteration spaces are used. this simply recreates MS0 in
// dawn/gtclang/test/integration-test/PassSetCaches/KCacheTest02.cpp, except that the two do methods
// are sepearted into their own stages
TEST(TestCaching, test_global_iteration_space_01) {
  using namespace dawn::iir;

  std::string stencilName("kcache");

  CartesianIIRBuilder b;
  auto out = b.field("out_field", FieldType::ijk);
  auto in = b.field("in_field", FieldType::ijk);
  auto tmp = b.tmpField("tmp", FieldType::ijk);

  auto stencilInstantiationContext =
      b.build(stencilName.c_str(),
              b.stencil(b.multistage(
                  dawn::iir::LoopOrderKind::Forward,
                  b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::Start,
                                     b.stmt(b.assignExpr(b.at(tmp), b.at(in))))),
                  b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End, 1, 0,
                                     b.stmt(b.assignExpr(b.at(out), b.at(tmp, {0, 0, -1}))))))));
  auto stencil = stencilInstantiationContext.at(stencilName.c_str());

  // dummy options
  Options compileOptions;
  OptimizerContext::OptimizerContextOptions optimizerOptions;

  // optimizer and compiler (required to run caching pass)
  DawnCompiler compiler(&compileOptions);
  OptimizerContext optimizer(compiler.getDiagnostics(), optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Cartesian));

  // run single compiler pass (caching)
  PassSetCaches cachingPass(optimizer);
  cachingPass.run(stencil);

  // check if the cache was set correctly (we expect a k cache fill on tmp)
  ASSERT_TRUE(stencil->getStencils().size() ==
              1); // we expect the instantiation to contain one stencil...
  ASSERT_TRUE(stencil->getStencils().at(0)->getChildren().size() == 1); //... with one multistage

  const auto& multiStage = *stencil->getStencils().at(0)->getChildren().begin();
  ASSERT_TRUE(multiStage->isCached(tmp.id));
  ASSERT_TRUE(multiStage->getCache(tmp.id).getIOPolicy() == iir::Cache::IOPolicy::fill);
  ASSERT_TRUE(multiStage->getCache(tmp.id).getType() == iir::Cache::CacheType::K);

  //   std::ofstream of("check_kcache.cpp");
  //   dump_cuda<dawn::codegen::cuda::CudaCodeGen>(of, stencil_instantiation);
  //   dump<dawn::codegen::cxxnaive::CXXNaiveCodeGen>(of, stencilInstantiationContext);
}

} // namespace