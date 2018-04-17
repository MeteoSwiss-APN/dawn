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

using namespace dawn;

namespace {

class ComputeStageExtents : public ::testing::Test {
  std::unique_ptr<dawn::Options> compileOptions_;

  dawn::DawnCompiler compiler_;

protected:
  ComputeStageExtents() : compiler_(compileOptions_.get()) {}
  virtual void SetUp() {}

  std::vector<std::shared_ptr<Stencil>> loadTest(std::string sirFilename) {

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

    return optimizer->getStencilInstantiationMap()["compute_extent_test_stencil"]->getStencils();
  }
};

TEST_F(ComputeStageExtents, test_stencil_01) {
  auto stencils = loadTest("compute_extent_test_stencil_01.sir");
  ASSERT_TRUE((stencils.size() == 1));
  std::shared_ptr<Stencil> stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 2));
  ASSERT_TRUE((stencil->getStage(0)->getExtents() == Extents{-1, 1, -1, 1, 0, 0}));
  ASSERT_TRUE((stencil->getStage(1)->getExtents() == Extents{0, 0, 0, 0, 0, 0}));
}

TEST_F(ComputeStageExtents, test_stencil_02) {
  auto stencils = loadTest("compute_extent_test_stencil_02.sir");

  ASSERT_TRUE((stencils.size() == 1));
  std::shared_ptr<Stencil> stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 3));
  ASSERT_TRUE((stencil->getStage(0)->getExtents() == Extents{-1, 1, -1, 1, 0, 0}));
  ASSERT_TRUE((stencil->getStage(1)->getExtents() == Extents{-1, 0, -1, 0, 0, 0}));
  ASSERT_TRUE((stencil->getStage(2)->getExtents() == Extents{0, 0, 0, 0, 0, 0}));
}
TEST_F(ComputeStageExtents, test_stencil_03) {
  auto stencils = loadTest("compute_extent_test_stencil_03.sir");
  ASSERT_TRUE((stencils.size() == 1));
  std::shared_ptr<Stencil> stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 4));
  ASSERT_TRUE((stencil->getStage(0)->getExtents() == Extents{-1, 1, -1, 2, 0, 0}));
  ASSERT_TRUE((stencil->getStage(1)->getExtents() == Extents{-1, 0, -1, 1, 0, 0}));
  ASSERT_TRUE((stencil->getStage(2)->getExtents() == Extents{0, 0, 0, 1, 0, 0}));
  ASSERT_TRUE((stencil->getStage(3)->getExtents() == Extents{0, 0, 0, 0, 0, 0}));
}

TEST_F(ComputeStageExtents, test_stencil_04) {
  auto stencils = loadTest("compute_extent_test_stencil_04.sir");

  ASSERT_TRUE((stencils.size() == 1));
  std::shared_ptr<Stencil> stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 4));
  ASSERT_TRUE((stencil->getStage(0)->getExtents() == Extents{-2, 3, -2, 1, 0, 0}));
  ASSERT_TRUE((stencil->getStage(1)->getExtents() == Extents{-1, 1, -1, 0, 0, 0}));
  ASSERT_TRUE((stencil->getStage(2)->getExtents() == Extents{0, 0, -1, 0, 0, 0}));
  ASSERT_TRUE((stencil->getStage(3)->getExtents() == Extents{0, 0, 0, 0, 0, 0}));
}

TEST_F(ComputeStageExtents, test_stencil_05) {
  auto stencils = loadTest("compute_extent_test_stencil_05.sir");
  ASSERT_TRUE((stencils.size() == 1));
  std::shared_ptr<Stencil> stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 4));
  ASSERT_TRUE((stencil->getStage(0)->getExtents() == Extents{-2, 3, -2, 1, 0, 0}));
  ASSERT_TRUE((stencil->getStage(1)->getExtents() == Extents{-1, 1, -1, 0, 0, 0}));
  ASSERT_TRUE((stencil->getStage(2)->getExtents() == Extents{0, 1, -1, 0, 0, 0}));
  ASSERT_TRUE((stencil->getStage(3)->getExtents() == Extents{0, 0, 0, 0, 0, 0}));
}

} // anonymous namespace
