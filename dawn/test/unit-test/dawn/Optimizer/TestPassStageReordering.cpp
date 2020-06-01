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

#include "dawn/IIR/IIR.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/PassMultiStageMerger.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassSetDependencyGraph.h"
#include "dawn/Optimizer/PassSetStageGraph.h"
#include "dawn/Optimizer/PassStageReordering.h"
#include "dawn/Serialization/IIRSerializer.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

class TestPassStageReordering : public ::testing::Test {
protected:
  OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_;

  explicit TestPassStageReordering() {
    context_ = std::make_unique<OptimizerContext>(options_,
                                                  std::make_shared<SIR>(ast::GridType::Cartesian));
    UIDGenerator::getInstance()->reset();
  }

  void runTest(const std::string& filename, const std::vector<unsigned>& stageOrders) {
    auto instantiation = IIRSerializer::deserialize(filename);

    // Run stage graph pass
    PassSetStageGraph stageGraphPass(*context_);
    EXPECT_TRUE(stageGraphPass.run(instantiation));

    // Run dependency graph pass
    PassSetDependencyGraph dependencyGraphPass(*context_);
    EXPECT_TRUE(dependencyGraphPass.run(instantiation));

    // Run multistage merge pass
    PassMultiStageMerger multiStageMerger(*context_);
    EXPECT_TRUE(multiStageMerger.run(instantiation));

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

TEST_F(TestPassStageReordering, ReorderTest3) {
  /*
     vertical_region(k_end - 1, k_start + 1) {
       field_b1 = field_b0;
       field_b2 = field_b1(k - 1);
     }
   */
  runTest("input/ReorderTest03.iir", {0, 1});
}

TEST_F(TestPassStageReordering, ReorderTest4) {
  /*
     vertical_region(k_end - 1, k_start + 1) {
       field_b1 = field_b0;
       field_b2 = field_b1(k - 1);  }
     vertical_region(k_start + 1, k_end - 1) {
       field_a1 = field_a0;
       field_a2 = field_a1(k + 1);  }
   */
  runTest("input/ReorderTest04.iir", {2, 0, 1, 3});
}

TEST_F(TestPassStageReordering, ReorderTest5) {
  /*
   vertical_region(k_start, k_start) { field_a1 = field_a0(k + 1); }
   vertical_region(k_start + 2, k_end - 1) { field_a2 = field_a1(k + 1); }
   */
  runTest("input/ReorderTest05.iir", {1, 0});
}

TEST_F(TestPassStageReordering, ReorderTest6) {
  /*
   vertical_region(k_start + 1, k_start + 1) {
     field_a1 = field_a0(k + 1);
     field_b1 = field_a0(k - 1);  }
   vertical_region(k_start + 3, k_end - 1) {
     field_a2 = field_a1(k + 1);
     field_b2 = field_b1(k - 1);  }
   */
  runTest("input/ReorderTest06.iir", {0, 1});
}

TEST_F(TestPassStageReordering, ReorderTest7) {
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
  runTest("input/ReorderTest07.iir", {0, 1, 2, 3, 4, 5, 6});
}

TEST_F(TestPassStageReordering, ReorderTest8) {
  /*vertical_region(k_start, k_start) {
      c = c / b;
      d = d / b;
    }
    vertical_region(k_start + 1, k_end) {
      double m = 1.0 / (b - a * c[k - 1]);
      c = c * m;
      d = (d - a * d[k - 1]) * m;
    }
    vertical_region(k_end - 1, k_start) {
      d -= c * d[k + 1];
    } */
  runTest("input/tridiagonal_solve.iir", {1, 0, 2, 4, 3, 5});
}

} // anonymous namespace
