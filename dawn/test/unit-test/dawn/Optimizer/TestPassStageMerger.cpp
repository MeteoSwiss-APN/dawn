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
#include "dawn/Optimizer/PassSetDependencyGraph.h"
#include "dawn/Optimizer/PassSetStageGraph.h"
#include "dawn/Optimizer/PassStageMerger.h"
#include "dawn/Serialization/IIRSerializer.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

class TestPassStageMerger : public ::testing::Test {
  dawn::Options options_;

protected:
  explicit TestPassStageMerger() : options_() {
    options_.MergeStages = options_.MergeDoMethods = true;
    UIDGenerator::getInstance()->reset();
  }

  void runTest(const std::string& filename, unsigned nStencils,
               const std::vector<unsigned>& nMultiStages, const std::vector<unsigned>& nStages,
               const std::vector<unsigned>& nDoMethods) {
    // Deserialize IIR
    auto instantiation = IIRSerializer::deserialize(filename);

    // Run stage graph pass
    PassSetStageGraph stageGraphPass;
    EXPECT_TRUE(stageGraphPass.run(instantiation, options_));

    // Run dependency graph pass
    PassSetDependencyGraph dependencyGraphPass;
    EXPECT_TRUE(dependencyGraphPass.run(instantiation, options_));

    // Expect pass to succeed...
    PassStageMerger stageMergerPass;
    EXPECT_TRUE(stageMergerPass.run(instantiation, options_));

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
  /*
    vertical_region(k_start, k_end) {
      field_a1 = field_a0;
    }
    vertical_region(k_start, k_end) {
      field_b1 = field_b0;
    } */
  runTest("input/StageMergerTest01.iir", 1, {1}, {1}, {1});
}

TEST_F(TestPassStageMerger, MergerTest2) {
  /*
    vertical_region(k_start, k_end) {
      field_a1 = field_a0;
    }
    vertical_region(k_start + 1, k_end) {
      field_b1 = field_b0;
    } */
  runTest("input/StageMergerTest02.iir", 1, {1}, {2}, {1, 1});
}

TEST_F(TestPassStageMerger, MergerTest3) {
  /*
    vertical_region(k_start, k_start) {
      field_a1 = field_a0;
    }
    vertical_region(k_start + 1, k_end) {
      field_b1 = field_b0;
    } */
  runTest("input/StageMergerTest03.iir", 1, {1}, {1}, {2});
}

TEST_F(TestPassStageMerger, MergerTest4) {
  /*
    vertical_region(k_start, k_start) {
      field_a1 = field_a0;
    }
    vertical_region(k_start + 1, k_end - 1) {
      field_a1 = field_a0;
    }
    vertical_region(k_end, k_end) {
      field_a1 = field_a0;
    } */
  runTest("input/StageMergerTest04.iir", 1, {1}, {1}, {3});
}

TEST_F(TestPassStageMerger, MergerTest5) {
  /*
    vertical_region(k_start, k_end) {
      field_a1 = field_a0;
    }
    vertical_region(k_start, k_end) {
      field_a2 = field_a1(i + 1);
    } */
  runTest("input/StageMergerTest05.iir", 1, {1}, {2}, {1, 1});
}

TEST_F(TestPassStageMerger, MergerTest6) {
  /*
    vertical_region(k_start, k_start) {
      field_a1 = field_a0;
    }
    vertical_region(k_start + 1, k_end) {
      field_a2 = field_a1(i + 1);
    } */
  runTest("input/StageMergerTest06.iir", 1, {1}, {1}, {2});
}

TEST_F(TestPassStageMerger, MergerTest7) {
  /*
    vertical_region(k_start, k_end) {
      out = in;
    }
    vertical_region(k_start, k_end) {
      out = 0;
    } */
  runTest("input/StageMergerTest07.iir", 1, {1}, {1}, {1});
}

TEST_F(TestPassStageMerger, MergerTestTwoCopies) {
  /*
    out_cells_1 = in_cells_1;
    out_cells_2 = in_cells_2;
   */
  runTest("input/StageMergerTestTwoCopies.iir", 1, {1}, {1}, {1});
}

TEST_F(TestPassStageMerger, MergerTestTwoCopiesMixed) {
  /*
    out_cells = in_cells;
    out_edges = in_edges;
   */
  runTest("input/StageMergerTestTwoCopiesMixed.iir", 1, {1}, {2}, {1, 1});
}

TEST_F(TestPassStageMerger, MergerTestUnstrIndependent) {
  // here, the stage sandwiched in the middle is independent from the stage
  // on edges above and below. we thus expect the stage merger to pass through
  // and merge eout1 = .... into eout= = ....
  //   eout0 = sum_over(Edge > Cell > Edge, ein0[Edge > Cell > Edge])
  //   cout0 = sum_over(Cell > Edge > Cell, cin0[Cell > Edge > Cell])
  //   eout1 = sum_over(Edge > Cell > Edge, ein1[Edge > Cell > Edge])
  runTest("input/StageMergerTestIndependent.iir", 1, {1}, {2}, {1, 1});
}

TEST_F(TestPassStageMerger, MergerTestUnstrDependent) {
  // here, the stage sandwiched in the middle has a dependency to the stage below
  // the stage merger hence must break and not try to merge eout1 = .... into eout= = ....
  //   eout0 = sum_over(Edge > Cell > Edge, ein0[Edge > Cell > Edge])
  //   cout0 = sum_over(Cell > Edge > Cell, cin0[Cell > Edge > Cell])
  //   eout1 = sum_over(Edge > Cell, cout0)
  runTest("input/StageMergerTestDependent.iir", 1, {1}, {3}, {1, 1, 1});
}

} // anonymous namespace
