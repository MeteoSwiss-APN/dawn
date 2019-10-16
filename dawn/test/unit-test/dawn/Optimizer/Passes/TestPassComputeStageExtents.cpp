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
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/SIRSerializer.h"
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

  std::unique_ptr<iir::IIR> loadTest(std::string sirFilename) {

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

    std::unique_ptr<iir::IIR>& iir =
        optimizer->getStencilInstantiationMap()["compute_extent_test_stencil"]->getIIR();
    return std::move(iir);
  }
};

static void compareExtents(iir::Extents extents, const std::array<int, 6>& ref) {
  const auto& h_extents =
      dawn::iir::extent_cast<dawn::iir::CartesianExtent const&>(extents.horizontalExtent());
  const auto& v_extents = extents.verticalExtent();

  EXPECT_EQ(h_extents.iMinus(), ref[0]);
  EXPECT_EQ(h_extents.iPlus(), ref[1]);
  EXPECT_EQ(h_extents.jMinus(), ref[2]);
  EXPECT_EQ(h_extents.jPlus(), ref[3]);
  EXPECT_EQ(v_extents.minus(), ref[4]);
  EXPECT_EQ(v_extents.plus(), ref[5]);
}

TEST_F(ComputeStageExtents, test_stencil_01) {
  std::unique_ptr<iir::IIR> IIR = loadTest("compute_extent_test_stencil_01.sir");
  const auto& stencils = IIR->getChildren();

  EXPECT_EQ(stencils.size(), 1);
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getNumStages(), 2);
  compareExtents(stencil->getStage(0)->getExtents(), {-1, 1, -1, 1, 0, 0});
  compareExtents(stencil->getStage(1)->getExtents(), {0, 0, 0, 0, 0, 0});
}

TEST_F(ComputeStageExtents, test_stencil_02) {
  std::unique_ptr<iir::IIR> IIR = loadTest("compute_extent_test_stencil_02.sir");
  const auto& stencils = IIR->getChildren();

  EXPECT_EQ(stencils.size(), 1);
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getNumStages(), 3);
  compareExtents(stencil->getStage(0)->getExtents(), {-1, 1, -1, 1, 0, 0});
  compareExtents(stencil->getStage(1)->getExtents(), {-1, 0, -1, 0, 0, 0});
  compareExtents(stencil->getStage(2)->getExtents(), {0, 0, 0, 0, 0, 0});
}
TEST_F(ComputeStageExtents, test_stencil_03) {
  std::unique_ptr<iir::IIR> IIR = loadTest("compute_extent_test_stencil_03.sir");
  const auto& stencils = IIR->getChildren();
  EXPECT_EQ(stencils.size(), 1);
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getNumStages(), 4);
  compareExtents(stencil->getStage(0)->getExtents(), {-1, 1, -1, 2, 0, 0});
  compareExtents(stencil->getStage(1)->getExtents(), {-1, 0, -1, 1, 0, 0});
  compareExtents(stencil->getStage(2)->getExtents(), {0, 0, 0, 1, 0, 0});
  compareExtents(stencil->getStage(3)->getExtents(), {0, 0, 0, 0, 0, 0});
}

TEST_F(ComputeStageExtents, test_stencil_04) {
  std::unique_ptr<iir::IIR> IIR = loadTest("compute_extent_test_stencil_04.sir");
  const auto& stencils = IIR->getChildren();

  EXPECT_EQ(stencils.size(), 1);
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getNumStages(), 4);
  compareExtents(stencil->getStage(0)->getExtents(), {-2, 3, -2, 1, 0, 0});
  compareExtents(stencil->getStage(1)->getExtents(), {-1, 1, -1, 0, 0, 0});
  compareExtents(stencil->getStage(2)->getExtents(), {0, 0, -1, 0, 0, 0});
  compareExtents(stencil->getStage(3)->getExtents(), {0, 0, 0, 0, 0, 0});
}

TEST_F(ComputeStageExtents, test_stencil_05) {
  std::unique_ptr<iir::IIR> IIR = loadTest("compute_extent_test_stencil_05.sir");
  const auto& stencils = IIR->getChildren();
  ASSERT_TRUE((stencils.size() == 1));
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getNumStages(), 4);
  compareExtents(stencil->getStage(0)->getExtents(), {-2, 3, -2, 1, 0, 0});
  compareExtents(stencil->getStage(1)->getExtents(), {-1, 0, -1, 0, 0, 0});
  compareExtents(stencil->getStage(2)->getExtents(), {0, 1, -1, 0, 0, 0});
  compareExtents(stencil->getStage(3)->getExtents(), {0, 0, 0, 0, 0, 0});
}

} // anonymous namespace
