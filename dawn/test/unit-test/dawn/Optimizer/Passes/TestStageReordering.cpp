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
#include "dawn/Optimizer/PassStageReordering.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Unittest/CompilerUtil.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

class TestStageReordering : public ::testing::Test {
protected:
  dawn::OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_;

  virtual void SetUp() { options_.ReportPassStageReodering = true; }

  void runTest(const std::string& filename, unsigned nStencils,
               const std::vector<unsigned>& nMultiStages, const std::vector<unsigned>& nStages,
               const std::vector<unsigned>& nDoMethods) {
    dawn::UIDGenerator::getInstance()->reset();
    std::shared_ptr<iir::StencilInstantiation> instantiation =
        CompilerUtil::load(filename, options_, context_, TestEnvironment::path_);

    ASSERT_TRUE(CompilerUtil::runGroup(PassGroup::Parallel, context_, instantiation));
    // ASSERT_TRUE(CompilerUtil::runGroup(PassGroup::ReorderStages, context_, instantiation));

    // Expect pass to succeed...
    ASSERT_TRUE(CompilerUtil::runPass<dawn::PassStageReordering>(
        context_, instantiation, dawn::ReorderStrategy::Kind::Greedy));

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

TEST_F(TestStageReordering, ReorderTest1) { runTest("Test01.sir", 1, {1}, {1}, {1}); }

TEST_F(TestStageReordering, ReorderTest2) { runTest("Test02.sir", 1, {1}, {4}, {1,1,1,1}); }

// TEST_F(TestStageReordering, ReorderTest3) { runTest("StageMergerTest03.sir", 1, {1}, {1}, {2}); }
//
// TEST_F(TestStageReordering, ReorderTest4) { runTest("StageMergerTest04.sir", 1, {1}, {1}, {3}); }
//
// TEST_F(TestStageReordering, ReorderTest5) { runTest("StageMergerTest05.sir", 1, {1}, {2}, {1,
// 1}); }
//
// TEST_F(TestStageReordering, ReorderTest6) { runTest("StageMergerTest06.sir", 1, {1}, {1}, {2}); }
//
// TEST_F(TestStageReordering, ReorderTest7) { runTest("StageMergerTest07.sir", 1, {1}, {1}, {1}); }

} // anonymous namespace
