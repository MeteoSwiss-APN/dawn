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
#include "dawn/Optimizer/PassMultiStageMerger.h"
#include "dawn/Optimizer/PassSetDependencyGraph.h"
#include "dawn/Optimizer/PassSetStageGraph.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Unittest/CompilerUtil.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

class TestPassMultiStageMerger : public ::testing::Test {
protected:
  dawn::OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_;
  dawn::DiagnosticsEngine diag_;

  explicit TestPassMultiStageMerger() {
    std::shared_ptr<SIR> sir = std::make_shared<SIR>(ast::GridType::Cartesian);
    context_ = std::make_unique<OptimizerContext>(diag_, options_, sir);
    dawn::UIDGenerator::getInstance()->reset();
  }

  void runTest(const std::string& filename, const std::vector<unsigned>& expNumStages) {
    std::string filepath = filename;
    if(!TestEnvironment::path_.empty())
      filepath = TestEnvironment::path_ + "/" + filepath;
    auto instantiation = IIRSerializer::deserialize(filepath);

    // Run stage graph pass
    PassSetStageGraph stageGraphPass(*context_);
    EXPECT_TRUE(stageGraphPass.run(instantiation));

    // Run dependency graph pass
    PassSetDependencyGraph dependencyGraphPass(*context_);
    EXPECT_TRUE(dependencyGraphPass.run(instantiation));

    // Expect pass to succeed...
    PassMultiStageMerger multiStageMerger(*context_);
    EXPECT_TRUE(multiStageMerger.run(instantiation));

    // Collect post-merging multi-stage counts
    std::vector<int> postNumStages;
    for(const auto& stencil : instantiation->getStencils())
      for(const auto& multistage : stencil->getChildren())
        postNumStages.push_back(multistage->getChildren().size());

    ASSERT_EQ(expNumStages.size(), postNumStages.size());
    for(int i = 0; i < expNumStages.size(); i++) {
      ASSERT_EQ(postNumStages[i], expNumStages[i]);
    }
  }
};

TEST_F(TestPassMultiStageMerger, MultiStageMergeTest1) {
  /*
     vertical_region(k_end, k_start) { field_a1 = field_a0; }
   */
  runTest("input/ReorderTest01.iir", {1});
}

TEST_F(TestPassMultiStageMerger, MultiStageMergeTest2) {
  /*
    vertical_region(k_end, k_start) { field_b1 = field_b0; }
    vertical_region(k_start, k_end) { field_a1 = field_a0; }
    vertical_region(k_end, k_start) { field_b2 = field_b1; }
    vertical_region(k_start, k_end) { field_a2 = field_a1; }
   */
  runTest("input/ReorderTest02.iir", {4});
}

TEST_F(TestPassMultiStageMerger, MultiStageMergeTest3) {
  /*
     vertical_region(k_end - 1, k_start + 1) {
       field_b1 = field_b0;
       field_b2 = field_b1(k - 1);
     } */
  runTest("input/ReorderTest03.iir", {2});
}

TEST_F(TestPassMultiStageMerger, MultiStageMergeTest4) {
  /*
     vertical_region(k_end - 1, k_start + 1) {
       field_b1 = field_b0;
       field_b2 = field_b1(k - 1);  }
     vertical_region(k_start + 1, k_end - 1) {
       field_a1 = field_a0;
       field_a2 = field_a1(k + 1);  }
   */
  runTest("input/ReorderTest04.iir", {3, 1});
}

TEST_F(TestPassMultiStageMerger, MultiStageMergeTest5) {
  /*
   vertical_region(k_start, k_start) { field_a1 = field_a0(k + 1); }
   vertical_region(k_start + 2, k_end - 1) { field_a2 = field_a1(k + 1); }
   */
  runTest("input/ReorderTest05.iir", {2});
}

TEST_F(TestPassMultiStageMerger, MultiStageMergeTest6) {
  /*
   vertical_region(k_start + 1, k_start + 1) {
     field_a1 = field_a0(k + 1);
     field_b1 = field_a0(k - 1);  }
   vertical_region(k_start + 3, k_end - 1) {
     field_a2 = field_a1(k + 1);
     field_b2 = field_b1(k - 1);  }
   */
  runTest("input/ReorderTest06.iir", {1, 1});
}

TEST_F(TestPassMultiStageMerger, MultiStageMergeTest7) {
  /*
    vertical_region(k_start, k_end) {
      field_a1 = field_a0(i + 1);
      field_a2 = field_a1(i + 1);
      field_a3 = field_a2(i + 1);
      field_a4 = field_a3(i + 1);
      field_a5 = field_a4(i + 1);
      field_a6 = field_a5(i + 1);
      field_a7 = field_a6(i + 1);
    }
   */
  runTest("input/ReorderTest07.iir", {7});
}

} // anonymous namespace
