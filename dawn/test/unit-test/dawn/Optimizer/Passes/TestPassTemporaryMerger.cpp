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
#include "dawn/Optimizer/PassTemporaryMerger.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Unittest/CompilerUtil.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

class TestPassTemporaryMerger : public ::testing::Test {
protected:
  dawn::OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_;

  virtual void SetUp() {
    options_.MergeTemporaries = options_.ReportPassTemporaryMerger = true;
  }

  void runTest(const std::string& filename, const std::vector<unsigned>& stageOrders) {
    dawn::UIDGenerator::getInstance()->reset();
    std::shared_ptr<iir::StencilInstantiation> instantiation =
        CompilerUtil::load(filename, options_, context_, TestEnvironment::path_);

    // Run parallel group
    CompilerUtil::Verbose = true;
    //ASSERT_TRUE(CompilerUtil::runPasses(12, context_, instantiation));

    ASSERT_TRUE(CompilerUtil::runGroup(PassGroup::Parallel, context_, instantiation));
    ASSERT_TRUE(CompilerUtil::runGroup(PassGroup::ReorderStages, context_, instantiation));
    ASSERT_TRUE(CompilerUtil::runGroup(PassGroup::MergeStages, context_, instantiation));

    // Expect pass to succeed...
    ASSERT_TRUE(CompilerUtil::runPass<dawn::PassTemporaryMerger>(context_, instantiation));

    for(const auto& stencil : instantiation->getStencils()) {
      for(const auto& multiStage : stencil->getChildren()) {
        for(const auto& stage : multiStage->getChildren()) {
          for(const auto& doMethod : stage->getChildren()) {
            int stop = 1;
          }
        }
      }
    }
  }
};

TEST_F(TestPassTemporaryMerger, MergeTest1) { runTest("MergeTest01.sir", {0}); }

TEST_F(TestPassTemporaryMerger, MergeTest2) { runTest("MergeTest02.sir", {1, 3, 0, 2}); }

TEST_F(TestPassTemporaryMerger, MergeTest3) { runTest("MergeTest03.sir", {0, 1}); }

//TEST_F(TestPassTemporaryMerger, MergeTest4) { runTest("MergeTest04.sir", {2, 0, 1, 3}); }
//
//TEST_F(TestPassTemporaryMerger, MergeTest5) { runTest("MergeTest05.sir", {1, 0}); }

} // anonymous namespace
