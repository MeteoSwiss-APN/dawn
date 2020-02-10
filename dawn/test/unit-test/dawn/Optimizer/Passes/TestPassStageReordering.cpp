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
#include "dawn/Optimizer/PassSetStageGraph.h"
#include "dawn/Optimizer/PassStageReordering.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Unittest/CompilerUtil.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

class TestPassStageReordering : public ::testing::Test {
protected:
  dawn::OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_;

  void runTest(const std::string& filename, const std::vector<unsigned>& stageOrders) {
    dawn::UIDGenerator::getInstance()->reset();
    std::shared_ptr<iir::StencilInstantiation> instantiation =
        CompilerUtil::load(filename, options_, context_, TestEnvironment::path_);

    // Run parallel group
    ASSERT_TRUE(CompilerUtil::runGroup(PassGroup::Parallel, context_, instantiation));

    // Collect pre-reordering stage IDs
    std::vector<int> prevStageIDs;
    for(const auto& stencil : instantiation->getStencils()) {
      for(const auto& multiStage : stencil->getChildren()) {
        for(const auto& stage : multiStage->getChildren()) {
          prevStageIDs.push_back(stage->getStageID());
        }
      }
    }

    int nStages = stageOrders.size();
    ASSERT_EQ(nStages, prevStageIDs.size());

    // Run stage graph pass
    ASSERT_TRUE(CompilerUtil::runPass<dawn::PassSetStageGraph>(context_, instantiation));

    // Expect pass to succeed...
    ASSERT_TRUE(CompilerUtil::runPass<dawn::PassStageReordering>(
        context_, instantiation, dawn::ReorderStrategy::Kind::Greedy));

    // Collect post-reordering stage IDs
    std::vector<int> postStageIDs;
    for(const auto& stencil : instantiation->getStencils()) {
      for(const auto& multiStage : stencil->getChildren()) {
        for(const auto& stage : multiStage->getChildren()) {
          postStageIDs.push_back(stage->getStageID());
        }
      }
    }

    ASSERT_EQ(nStages, postStageIDs.size());
    for(int i = 0; i < nStages; i++) {
      ASSERT_EQ(postStageIDs[i], prevStageIDs[stageOrders[i]]);
    }
  }
};

TEST_F(TestPassStageReordering, ReorderTest1) { runTest("input/ReorderTest01.sir", {0}); }

TEST_F(TestPassStageReordering, ReorderTest2) { runTest("input/ReorderTest02.sir", {1, 3, 0, 2}); }

TEST_F(TestPassStageReordering, ReorderTest3) { runTest("input/ReorderTest03.sir", {0, 1}); }

TEST_F(TestPassStageReordering, ReorderTest4) { runTest("input/ReorderTest04.sir", {2, 0, 1, 3}); }

TEST_F(TestPassStageReordering, ReorderTest5) { runTest("input/ReorderTest05.sir", {1, 0}); }

TEST_F(TestPassStageReordering, ReorderTest6) { runTest("input/ReorderTest06.sir", {0, 1}); }

TEST_F(TestPassStageReordering, ReorderTest7) {
  runTest("input/ReorderTest07.sir", {0, 1, 2, 3, 4, 5, 6});
}

} // anonymous namespace
