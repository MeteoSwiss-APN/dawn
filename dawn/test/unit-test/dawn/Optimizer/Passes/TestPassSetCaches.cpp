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
#include "dawn/Optimizer/PassSetCaches.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Unittest/CompilerUtil.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

class TestPassSetCaches : public ::testing::Test {
protected:
  dawn::OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_;

  void runTest(const std::string& filename, int nStencils, int nMultiStages,
               const std::vector<std::vector<std::string>>& fieldNames,
               const std::vector<std::vector<iir::Cache::CacheType>>& cacheTypes,
               const std::vector<std::vector<iir::Cache::IOPolicy>>& ioPolicies) {
    if(nStencils < 1)
      nStencils = 1;

    std::shared_ptr<iir::StencilInstantiation> instantiation =
        CompilerUtil::load(filename, options_, context_, TestEnvironment::path_);

    ASSERT_TRUE(CompilerUtil::runGroup(PassGroup::Parallel, context_, instantiation));
    ASSERT_TRUE(CompilerUtil::runGroup(PassGroup::StageReordering, context_, instantiation));

    // Expect pass to succeed...
    ASSERT_TRUE(CompilerUtil::runPass<dawn::PassSetCaches>(context_, instantiation));

    auto& stencils = instantiation->getStencils();
    ASSERT_EQ(stencils.size(), nStencils);

    for(int i = 0; i < nStencils; ++i) {
      auto& multiStages = stencils[i]->getChildren();
      ASSERT_EQ(multiStages.size(), nMultiStages);

      int j = 0;
      for(const auto& multiStage : multiStages) {
        unsigned nFields = fieldNames[j].size();
        for(int k = 0; k < nFields; ++k) {
          int accessID = stencils[i]->getMetadata().getAccessIDFromName(fieldNames[j][k]);
          ASSERT_TRUE(multiStage->isCached(accessID));
          ASSERT_TRUE(multiStage->getCache(accessID).getType() == cacheTypes[j][k]);
          ASSERT_TRUE(multiStage->getCache(accessID).getIOPolicy() == ioPolicies[j][k]);
        }
        j += 1;
      }
    }
  }
};

TEST_F(TestPassSetCaches, IJCacheTest1) {
  runTest("input/IJCacheTest01.sir", 1, 1, {{"tmp"}}, {{iir::Cache::CacheType::IJ}},
          {{iir::Cache::IOPolicy::local}});
}

TEST_F(TestPassSetCaches, IJCacheTest2) {
  runTest("input/IJCacheTest02.sir", 1, 1, {{"tmp"}}, {{iir::Cache::CacheType::IJ}},
          {{iir::Cache::IOPolicy::local}});
}

TEST_F(TestPassSetCaches, KCacheTest1) {
  runTest("input/KCacheTest01.sir", 1, 1, {{"tmp"}}, {{iir::Cache::CacheType::K}},
          {{iir::Cache::IOPolicy::fill}});
}

TEST_F(TestPassSetCaches, KCacheTest1b) {
  runTest("input/KCacheTest01b.sir", 1, 1, {{"tmp"}}, {{iir::Cache::CacheType::K}},
          {{iir::Cache::IOPolicy::local}});
}

TEST_F(TestPassSetCaches, KCacheTest2) {
  runTest("input/KCacheTest02.sir", 1, 2, {{"tmp"}, {"tmp"}},
          {{iir::Cache::CacheType::K}, {iir::Cache::CacheType::K}},
          {{iir::Cache::IOPolicy::fill_and_flush}, {iir::Cache::IOPolicy::fill}});
}

TEST_F(TestPassSetCaches, KCacheTest2b) {
  runTest("input/KCacheTest02b.sir", 1, 2, {{"tmp"}, {"b", "tmp"}},
          {{iir::Cache::CacheType::K}, {iir::Cache::CacheType::K, iir::Cache::CacheType::K}},
          {{iir::Cache::IOPolicy::fill_and_flush},
           {iir::Cache::IOPolicy::fill, iir::Cache::IOPolicy::bpfill}});
}

} // anonymous namespace
