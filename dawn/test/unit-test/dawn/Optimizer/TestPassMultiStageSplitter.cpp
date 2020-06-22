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
#include "dawn/IIR/LoopOrder.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/PassMultiStageSplitter.h"
#include "dawn/Serialization/IIRSerializer.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

class TestPassMultiStageSplitter : public ::testing::Test {
protected:
  std::shared_ptr<iir::StencilInstantiation> runPass(const std::string& filename) {
    auto instantiation = IIRSerializer::deserialize(filename);

    // Expect pass to succeed...
    auto mssSplitStrategy = dawn::PassMultiStageSplitter::MultiStageSplittingStrategy::Optimized;
    PassMultiStageSplitter splitter(mssSplitStrategy);
    splitter.run(instantiation);

    return instantiation;
  }
};

int getNumberOfMultistages(iir::StencilInstantiation& instantiation) {
  return instantiation.getIIR()->getChild(0)->getChildren().size();
}

TEST_F(TestPassMultiStageSplitter, SplitterTest1) {
  /*
    vertical_region(k_start, k_end) { field_a = field_b; }
  */
  auto instantiation = runPass("input/TestMultiStageSplitter_01.iir");
  ASSERT_EQ(getNumberOfMultistages(*instantiation), 1);
  auto& multiStage = instantiation->getIIR()->getChild(0)->getChild(0);
  ASSERT_EQ(multiStage->getLoopOrder(), iir::LoopOrderKind::Parallel);
}

TEST_F(TestPassMultiStageSplitter, SplitterTest2) {
  /*
    vertical_region(k_start, k_end - 1) {
      field_b =field_c;
      field_a = field_b[k + 1];
    }
  */
  auto instantiation = runPass("input/TestMultiStageSplitter_02.iir");
  ASSERT_EQ(getNumberOfMultistages(*instantiation), 2);
  auto& multiStage0 = instantiation->getIIR()->getChild(0)->getChild(0);
  ASSERT_EQ(multiStage0->getLoopOrder(), iir::LoopOrderKind::Parallel);
  auto& multiStage1 = instantiation->getIIR()->getChild(0)->getChild(1);
  ASSERT_EQ(multiStage1->getLoopOrder(), iir::LoopOrderKind::Parallel);
}

TEST_F(TestPassMultiStageSplitter, SplitterTest3) {
  /*
    vertical_region(k_end, k_start + 1) {
      field_b = field_c;
      field_a = field_b[k - 1];
    }
  */
  auto instantiation = runPass("input/TestMultiStageSplitter_03.iir");
  ASSERT_EQ(getNumberOfMultistages(*instantiation), 2);
  auto& multiStage0 = instantiation->getIIR()->getChild(0)->getChild(0);
  ASSERT_EQ(multiStage0->getLoopOrder(), iir::LoopOrderKind::Parallel);
  auto& multiStage1 = instantiation->getIIR()->getChild(0)->getChild(1);
  ASSERT_EQ(multiStage1->getLoopOrder(), iir::LoopOrderKind::Parallel);
}

TEST_F(TestPassMultiStageSplitter, SplitterTest4) {
  /*
    vertical_region(k_start, k_end - 1) {
      field_b = field_c;
      field_a = field_b[k + 1];
    }

    vertical_region(k_end - 1, k_start) {
      field_b = field_c;
      field_a = field_b[k - 1];
    }
  */
  auto instantiation = runPass("input/TestMultiStageSplitter_04.iir");
  ASSERT_EQ(getNumberOfMultistages(*instantiation), 4);
  for(auto& multiStage : instantiation->getIIR()->getChild(0)->getChildren()) {
    ASSERT_EQ(multiStage->getLoopOrder(), iir::LoopOrderKind::Parallel);
  }
}

TEST_F(TestPassMultiStageSplitter, SplitterTest5) {
  /*
    vertical_region(k_start + 1, k_end - 1) {
      field_c = field_d;
      field_b = field_c[k + 1];
      field_a = field_b[k - 1];
    }
  */
  auto instantiation = runPass("input/TestMultiStageSplitter_05.iir");
  ASSERT_EQ(getNumberOfMultistages(*instantiation), 2);
  auto& multiStage0 = instantiation->getIIR()->getChild(0)->getChild(0);
  ASSERT_EQ(multiStage0->getLoopOrder(), iir::LoopOrderKind::Parallel);
  auto& multiStage1 = instantiation->getIIR()->getChild(0)->getChild(1);
  ASSERT_EQ(multiStage1->getLoopOrder(), iir::LoopOrderKind::Forward);
  ASSERT_TRUE(true);
}

} // anonymous namespace
