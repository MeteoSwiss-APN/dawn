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
#include "dawn/Optimizer/PassMultiStageSplitter.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Unittest/CompilerUtil.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

class TestMultiStageSplitter : public ::testing::Test {
protected:
  dawn::OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_;

  void runTest(const std::string& filename, int nStencils, const std::vector<int>& nMultiStages) {
    if(nStencils < 1)
      nStencils = 1;

    std::shared_ptr<iir::StencilInstantiation> instantiation =
        CompilerUtil::load(filename, options_, context_, TestEnvironment::path_);

    // Expect pass to succeed...
    auto mssSplitStrategy = dawn::PassMultiStageSplitter::MultiStageSplittingStrategy::Optimized;
    ASSERT_TRUE(CompilerUtil::runPass<dawn::PassMultiStageSplitter>(context_, instantiation,
                                                                    mssSplitStrategy));

    auto& stencils = instantiation->getStencils();
    ASSERT_EQ(stencils.size(), nStencils);

    for(int i = 0; i < nStencils; i++) {
      ASSERT_EQ(stencils[i]->getChildren().size(), nMultiStages[i]);
    }
  }
};

TEST_F(TestMultiStageSplitter, SplitterTest1) { runTest("SplitterTest01.sir", 1, {1}); }

TEST_F(TestMultiStageSplitter, SplitterTest2) { runTest("SplitterTest02.sir", 1, {2}); }

TEST_F(TestMultiStageSplitter, SplitterTest3) { runTest("SplitterTest03.sir", 1, {2}); }

TEST_F(TestMultiStageSplitter, SplitterTest4) { runTest("SplitterTest04.sir", 1, {4}); }

TEST_F(TestMultiStageSplitter, SplitterTest5) { runTest("SplitterTest05.sir", 1, {2}); }

} // anonymous namespace
