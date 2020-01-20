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
  DawnCompiler compiler(compileOptions);
  OptimizerContext optimizer(compiler.getDiagnostics(), optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Cartesian));

  // run single compiler pass (caching)
  PassSetCaches cachingPass(optimizer);
  cachingPass.run(stencil);

  ASSERT_TRUE(stencil->getStencils().size() ==
              1); // we expect the instantiation to contain one stencil...
  ASSERT_TRUE(stencil->getStencils().at(0)->getChildren().size() == 1); //... with one multistage

  // check if the cache was set correctly (we expect a k cache fill on tmp)
  const auto& multiStage = *stencil->getStencils().at(0)->getChildren().begin();
  ASSERT_TRUE(multiStage->isCached(tmp.id));
  ASSERT_TRUE(multiStage->getCache(tmp.id).getIOPolicy() == iir::Cache::IOPolicy::fill);
  ASSERT_TRUE(multiStage->getCache(tmp.id).getType() == iir::Cache::CacheType::K);
}

// first stage with iteration space, second stage with compatible iteration space
TEST(TestCaching, test_global_iteration_space_02) {
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
                  b.stage(Interval(0, 10), Interval(0, 10),
                          b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::Start,
                                     b.stmt(b.assignExpr(b.at(tmp), b.at(in))))),
                  b.stage(Interval(2, 8), Interval(3, 10),
                          b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End, 1, 0,
                                     b.stmt(b.assignExpr(b.at(out), b.at(tmp, {0, 0, -1}))))))));
  auto stencil = stencilInstantiationContext.at(stencilName.c_str());

  // dummy options
  Options compileOptions;
  OptimizerContext::OptimizerContextOptions optimizerOptions;

  // optimizer and compiler (required to run caching pass)
  DawnCompiler compiler(compileOptions);
  OptimizerContext optimizer(compiler.getDiagnostics(), optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Cartesian));

  // run single compiler pass (caching)
  PassSetCaches cachingPass(optimizer);
  cachingPass.run(stencil);

  ASSERT_TRUE(stencil->getStencils().size() ==
              1); // we expect the instantiation to contain one stencil...
  ASSERT_TRUE(stencil->getStencils().at(0)->getChildren().size() == 1); //... with one multistage

  // check if the cache was set correctly (we expect a k cache fill on tmp since the iteration
  // spaces are compatible))
  const auto& multiStage = *stencil->getStencils().at(0)->getChildren().begin();
  ASSERT_TRUE(multiStage->isCached(tmp.id));
  ASSERT_TRUE(multiStage->getCache(tmp.id).getIOPolicy() == iir::Cache::IOPolicy::fill);
  ASSERT_TRUE(multiStage->getCache(tmp.id).getType() == iir::Cache::CacheType::K);
}

// first stage with iteration space, second stage with incompatible iteration space
TEST(TestCaching, test_global_iteration_space_03) {
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
                  b.stage(Interval(0, 10), Interval(0, 10),
                          b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::Start,
                                     b.stmt(b.assignExpr(b.at(tmp), b.at(in))))),
                  b.stage(Interval(2, 12), Interval(10, 13),
                          b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End, 1, 0,
                                     b.stmt(b.assignExpr(b.at(out), b.at(tmp, {0, 0, -1}))))))));
  auto stencil = stencilInstantiationContext.at(stencilName.c_str());

  // dummy options
  Options compileOptions;
  OptimizerContext::OptimizerContextOptions optimizerOptions;

  // optimizer and compiler (required to run caching pass)
  DawnCompiler compiler(compileOptions);
  OptimizerContext optimizer(compiler.getDiagnostics(), optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Cartesian));

  // run single compiler pass (caching)
  PassSetCaches cachingPass(optimizer);
  cachingPass.run(stencil);

  ASSERT_TRUE(stencil->getStencils().size() ==
              1); // we expect the instantiation to contain one stencil...
  ASSERT_TRUE(stencil->getStencils().at(0)->getChildren().size() == 1); //... with one multistage

  // we expect no caching in this case since the iteration spaces are not compatible
  const auto& multiStage = *stencil->getStencils().at(0)->getChildren().begin();
  ASSERT_TRUE(!multiStage->isCached(tmp.id));
}

// first stage with iteration space, second stage without iteration space
TEST(TestCaching, test_global_iteration_space_04) {
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
                  b.stage(Interval(0, 10), Interval(0, 10),
                          b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::Start,
                                     b.stmt(b.assignExpr(b.at(tmp), b.at(in))))),
                  b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End, 1, 0,
                                     b.stmt(b.assignExpr(b.at(out), b.at(tmp, {0, 0, -1}))))))));
  auto stencil = stencilInstantiationContext.at(stencilName.c_str());

  // dummy options
  Options compileOptions;
  OptimizerContext::OptimizerContextOptions optimizerOptions;

  // optimizer and compiler (required to run caching pass)
  DawnCompiler compiler(compileOptions);
  OptimizerContext optimizer(compiler.getDiagnostics(), optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Cartesian));

  // run single compiler pass (caching)
  PassSetCaches cachingPass(optimizer);
  cachingPass.run(stencil);

  ASSERT_TRUE(stencil->getStencils().size() ==
              1); // we expect the instantiation to contain one stencil...
  ASSERT_TRUE(stencil->getStencils().at(0)->getChildren().size() == 1); //... with one multistage

  // we expect no caching in this case since the iteration spaces are not compatible
  const auto& multiStage = *stencil->getStencils().at(0)->getChildren().begin();
  ASSERT_TRUE(!multiStage->isCached(tmp.id));
}

// first stage without iteration space, second stage iteration space
TEST(TestCaching, test_global_iteration_space_05) {
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
                  b.stage(Interval(0, 10), Interval(0, 10),
                          b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End, 1, 0,
                                     b.stmt(b.assignExpr(b.at(out), b.at(tmp, {0, 0, -1}))))))));
  auto stencil = stencilInstantiationContext.at(stencilName.c_str());

  // dummy options
  Options compileOptions;
  OptimizerContext::OptimizerContextOptions optimizerOptions;

  // optimizer and compiler (required to run caching pass)
  DawnCompiler compiler(compileOptions);
  OptimizerContext optimizer(compiler.getDiagnostics(), optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Cartesian));

  // run single compiler pass (caching)
  PassSetCaches cachingPass(optimizer);
  cachingPass.run(stencil);

  ASSERT_TRUE(stencil->getStencils().size() ==
              1); // we expect the instantiation to contain one stencil...
  ASSERT_TRUE(stencil->getStencils().at(0)->getChildren().size() == 1); //... with one multistage

  // check if the cache was set correctly (we expect a k cache fill on tmp since an iteration in the
  // second stage should not influence the caching)
  const auto& multiStage = *stencil->getStencils().at(0)->getChildren().begin();
  ASSERT_TRUE(multiStage->isCached(tmp.id));
  ASSERT_TRUE(multiStage->getCache(tmp.id).getIOPolicy() == iir::Cache::IOPolicy::fill);
  ASSERT_TRUE(multiStage->getCache(tmp.id).getType() == iir::Cache::CacheType::K);
}

} // namespace