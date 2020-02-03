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

  void runIJTest(const std::string& filename, int nStencils, const std::vector<int>& fieldIDs) {
    if(nStencils < 1)
      nStencils = 1;

    std::shared_ptr<iir::StencilInstantiation> instantiation;
    //options_.ReportPassSetCaches = true;
    CompilerUtil::load(filename, options_, context_, instantiation, TestEnvironment::path_);

    PassSetCaches pass(*context_);
    bool result = pass.run(instantiation);
    ASSERT_TRUE(result); // Expect pass to succeed...

    auto& stencils = instantiation->getStencils();
    ASSERT_EQ(stencils.size(), nStencils);

    for(int i = 0; i < nStencils; i++) {
      auto& multiStages = stencils[i]->getChildren();
      unsigned field_id = fieldIDs[i];
      for(const auto& multiStage : multiStages) {
        ASSERT_TRUE(multiStage->isCached(field_id));
        ASSERT_TRUE(multiStage->getCache(field_id).getType() == iir::Cache::CacheType::IJ);
        ASSERT_TRUE(multiStage->getCache(field_id).getIOPolicy() == iir::Cache::IOPolicy::local);
      }
    }
  }
};

TEST_F(TestPassSetCaches, IJCacheTest1) { runIJTest("IJCacheTest01.iir", 1, {3}); }

TEST_F(TestPassSetCaches, IJCacheTest2) { runIJTest("IJCacheTest02.iir", 1, {3}); }

} // anonymous namespace
