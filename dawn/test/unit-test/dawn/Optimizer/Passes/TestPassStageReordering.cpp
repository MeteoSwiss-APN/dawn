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
#include "dawn/Optimizer/PassStageSplitter.h"
#include "dawn/Optimizer/PassSetStageGraph.h"
#include "dawn/Optimizer/PassStageReordering.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

class TestPassStageReordering : public ::testing::Test {
protected:
  dawn::OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_;
  dawn::DiagnosticsEngine diag_;

  explicit TestPassStageReordering() {
    std::shared_ptr<SIR> sir = std::make_shared<SIR>(ast::GridType::Cartesian);
    context_ = std::make_unique<OptimizerContext>(diag_, options_, sir);
  }

  void runTest(const std::string& filename, const std::vector<unsigned>& stageOrders) {
    dawn::UIDGenerator::getInstance()->reset();
    auto instantiation = IIRSerializer::deserialize(filename);

    // Run stage splitter pass
    PassStageSplitter stageSplitPass(*context_);
    EXPECT_TRUE(stageSplitPass.run(instantiation));

    // Run stage graph pass
    PassSetStageGraph stageGraphPass(*context_);
    EXPECT_TRUE(stageGraphPass.run(instantiation));

    // Collect pre-reordering stage IDs
    std::vector<int> prevStageIDs;
    for(const auto& stencil : instantiation->getStencils())
      for(const auto& multiStage : stencil->getChildren())
        for(const auto& stage : multiStage->getChildren())
          prevStageIDs.push_back(stage->getStageID());

    EXPECT_EQ(stageOrders.size(), prevStageIDs.size());

    // Expect pass to succeed...
    PassStageReordering stageReorderPass(*context_, dawn::ReorderStrategy::Kind::Greedy);
    EXPECT_TRUE(stageReorderPass.run(instantiation));

    // Collect post-reordering stage IDs
    std::vector<int> postStageIDs;
    for(const auto& stencil : instantiation->getStencils())
      for(const auto& multiStage : stencil->getChildren())
        for(const auto& stage : multiStage->getChildren())
          postStageIDs.push_back(stage->getStageID());

    ASSERT_EQ(prevStageIDs.size(), postStageIDs.size());
    for(int i = 0; i < stageOrders.size(); i++) {
      ASSERT_EQ(postStageIDs[i], prevStageIDs[stageOrders[i]]);
    }
  }
};

TEST_F(TestPassStageReordering, ReorderTest1) {
  /*
     vertical_region(k_end, k_start) { field_a1 = field_a0; }
   */
  runTest("input/ReorderTest01.iir", {0});
}

TEST_F(TestPassStageReordering, ReorderTest2) {
  /*
    vertical_region(k_end, k_start) { field_b1 = field_b0; }
    vertical_region(k_start, k_end) { field_a1 = field_a0; }
    vertical_region(k_end, k_start) { field_b2 = field_b1; }
    vertical_region(k_start, k_end) { field_a2 = field_a1; }
   */
  runTest("input/ReorderTest02.iir", {1, 3, 0, 2});
}

//TEST_F(TestPassStageReordering, ReorderTest3) { runTest("input/ReorderTest03.sir", {0, 1}); }
//
//TEST_F(TestPassStageReordering, ReorderTest4) { runTest("input/ReorderTest04.sir", {2, 0, 1, 3}); }
//
//TEST_F(TestPassStageReordering, ReorderTest5) { runTest("input/ReorderTest05.sir", {1, 0}); }
//
//TEST_F(TestPassStageReordering, ReorderTest6) { runTest("input/ReorderTest06.sir", {0, 1}); }
//
//TEST_F(TestPassStageReordering, ReorderTest7) {
//  runTest("input/ReorderTest07.sir", {0, 1, 2, 3, 4, 5, 6});
//}

} // anonymous namespace
