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

#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/Compiler/Options.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/PassStageMerger.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Unittest/CompilerUtil.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

class TestPassStageMerger : public ::testing::Test {
protected:
  dawn::OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_;

  virtual void SetUp() { options_.MergeStages = options_.MergeDoMethods = true; }

  void runTest(const std::string& filename, unsigned nStencils,
               const std::vector<unsigned>& nMultiStages, const std::vector<unsigned>& nStages,
               const std::vector<unsigned>& nDoMethods) {
    dawn::UIDGenerator::getInstance()->reset();
    std::shared_ptr<iir::StencilInstantiation> instantiation =
        CompilerUtil::load(filename, options_, context_, TestEnvironment::path_);

    ASSERT_TRUE(CompilerUtil::runGroup(PassGroup::Parallel, context_, instantiation));
    ASSERT_TRUE(CompilerUtil::runGroup(PassGroup::ReorderStages, context_, instantiation));

    // Expect pass to succeed...
    ASSERT_TRUE(CompilerUtil::runPass<dawn::PassStageMerger>(context_, instantiation));

    unsigned stencilIdx = 0;
    unsigned msIdx = 0;
    unsigned stageIdx = 0;

    ASSERT_EQ(nStencils, instantiation->getStencils().size());
    for(const auto& stencil : instantiation->getStencils()) {
      ASSERT_EQ(nMultiStages[stencilIdx], stencil->getChildren().size());
      for(const auto& multiStage : stencil->getChildren()) {
        ASSERT_EQ(nStages[msIdx], multiStage->getChildren().size());
        for(const auto& stage : multiStage->getChildren()) {
          ASSERT_EQ(nDoMethods[stageIdx], stage->getChildren().size());
          stageIdx += 1;
        }
        msIdx += 1;
      }
      stencilIdx += 1;
    }
  }
};

TEST_F(TestPassStageMerger, MergerTest1) {
  runTest("input/StageMergerTest01.sir", 1, {1}, {1}, {1});
}

TEST_F(TestPassStageMerger, MergerTest2) {
  runTest("input/StageMergerTest02.sir", 1, {1}, {2}, {1, 1});
}

TEST_F(TestPassStageMerger, MergerTest3) {
  runTest("input/StageMergerTest03.sir", 1, {1}, {1}, {2});
}

TEST_F(TestPassStageMerger, MergerTest4) {
  runTest("input/StageMergerTest04.sir", 1, {1}, {1}, {3});
}

TEST_F(TestPassStageMerger, MergerTest5) {
  runTest("input/StageMergerTest05.sir", 1, {1}, {2}, {1, 1});
}

TEST_F(TestPassStageMerger, MergerTest6) {
  runTest("input/StageMergerTest06.sir", 1, {1}, {1}, {2});
}

TEST_F(TestPassStageMerger, MergerTest7) {
  runTest("input/StageMergerTest07.sir", 1, {1}, {1}, {1});
}

} // anonymous namespace
