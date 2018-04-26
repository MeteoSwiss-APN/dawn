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
#include "dawn/SIR/SIR.h"
#include "dawn/SIR/SIRSerializer.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"
#include <fstream>
#include <gtest/gtest.h>
#include <streambuf>
#include <string>

using namespace dawn;

namespace {

class TestComputeMaximumExtent : public ::testing::Test {
  std::unique_ptr<dawn::Options> compileOptions_;

  dawn::DawnCompiler compiler_;

protected:
  TestComputeMaximumExtent() : compiler_(compileOptions_.get()) {}
  virtual void SetUp() {}

  std::shared_ptr<StencilInstantiation> loadTest(std::string sirFilename) {

    std::string filename = TestEnvironment::path_ + "/" + sirFilename;
    std::ifstream file(filename);
    DAWN_ASSERT_MSG((file.good()), std::string("File " + filename + " does not exists").c_str());

    std::string jsonstr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::shared_ptr<SIR> sir =
        SIRSerializer::deserializeFromString(jsonstr, SIRSerializer::SK_Json);

    std::unique_ptr<OptimizerContext> optimizer = compiler_.runOptimizer(sir);
    // Report diganostics
    if(compiler_.getDiagnostics().hasDiags()) {
      for(const auto& diag : compiler_.getDiagnostics().getQueue())
        std::cerr << "Compilation Error " << diag->getMessage() << std::endl;
      throw std::runtime_error("compilation failed");
    }

    DAWN_ASSERT_MSG((optimizer->getStencilInstantiationMap().count("compute_extent_test_stencil")),
                    "compute_extent_test_stencil not found in sir");

    return optimizer->getStencilInstantiationMap()["compute_extent_test_stencil"];
  }
};

TEST_F(TestComputeMaximumExtent, test_field_access_interval_01) {
  auto stencilInstantiation = loadTest("test_field_access_interval_02.sir");
  auto stencils = stencilInstantiation->getStencils();
  ASSERT_TRUE((stencils.size() == 1));
  std::shared_ptr<Stencil> stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 2));
  ASSERT_TRUE((stencil->getStage(0)->getExtents() == Extents{-1, 1, -1, 1, 0, 0}));
  ASSERT_TRUE((stencil->getStage(1)->getExtents() == Extents{0, 0, 0, 0, 0, 0}));

  ASSERT_TRUE((stencil->getMultiStages().size() == 1));

  auto const& mss = stencil->getMultiStages().front();

  auto stage1_ptr = mss->getStages().begin();
  std::shared_ptr<Stage> const& stage1 = *stage1_ptr;

  ASSERT_TRUE((stage1->getDoMethods().size() == 2));

  const auto& doMethod1 = stage1->getDoMethods()[0];

  ASSERT_TRUE((doMethod1->getStatementAccessesPairs().size() == 1));
  const auto& stmtAccessPair = doMethod1->getStatementAccessesPairs()[0];
  ASSERT_TRUE((stmtAccessPair->computeMaximumExtents(
                   stencilInstantiation->getAccessIDFromName("u")) == Extents{-1, 1, -1, 1, 0, 0}));

  ASSERT_TRUE((stmtAccessPair->computeMaximumExtents(stencilInstantiation->getAccessIDFromName(
                   "coeff")) == Extents{0, 0, 0, 0, 0, 1}));
}

TEST_F(TestComputeMaximumExtent, test_compute_maximum_extent_01) {
  auto stencilInstantiation = loadTest("test_compute_maximum_extent_01.sir");
  auto stencils = stencilInstantiation->getStencils();

  ASSERT_TRUE((stencils.size() == 1));
  std::shared_ptr<Stencil> stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 1));
  ASSERT_TRUE((stencil->getMultiStages().size() == 1));

  auto const& mss = stencil->getMultiStages().front();

  auto stage1_ptr = mss->getStages().begin();
  std::shared_ptr<Stage> const& stage1 = *stage1_ptr;

  ASSERT_TRUE((stage1->getDoMethods().size() == 1));

  const auto& doMethod1 = stage1->getDoMethods()[0];

  ASSERT_TRUE((doMethod1->computeMaximumExtents(stencilInstantiation->getAccessIDFromName("u")) ==
               Extents{0, 1, -1, 0, 0, 2}));
}

} // anonymous namespace
