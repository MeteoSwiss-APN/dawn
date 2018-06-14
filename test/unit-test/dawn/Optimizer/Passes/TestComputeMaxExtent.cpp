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

class ComputeMaxExtents : public ::testing::Test {
  std::unique_ptr<dawn::Options> compileOptions_;

  dawn::DawnCompiler compiler_;

protected:
  ComputeMaxExtents() : compiler_(compileOptions_.get()) {}
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

TEST_F(ComputeMaxExtents, test_stencil_01) {
  auto stencils = loadTest("compute_extent_test_stencil_01.sir");
  ASSERT_TRUE((stencils.size() == 1));
  std::shared_ptr<Stencil> stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 2));
  auto exts = stencil->computeEnclosingAccessExtents();
  EXPECT_EQ(exts.size(), 3);
  EXPECT_EQ((stencil->getStencilInstantiation().getNameFromAccessID(1)), "u");
  EXPECT_EQ((stencil->getStencilInstantiation().getNameFromAccessID(2)), "out");
  EXPECT_EQ((stencil->getStencilInstantiation().getNameFromAccessID(3)), "lap");
  EXPECT_EQ(exts[1], (Extents{-2,2,-2,2,0,0}));
  EXPECT_EQ(exts[2], (Extents{0,0,0,0,0,0}));
  EXPECT_EQ(exts[3], (Extents{-1,1,-1,1,0,0}));
}

TEST_F(ComputeMaxExtents, test_stencil_02) {
  auto stencils = loadTest("compute_extent_test_stencil_02.sir");
  ASSERT_TRUE((stencils.size() == 1));
  std::shared_ptr<Stencil> stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 3));
  auto exts = stencil->computeEnclosingAccessExtents();
  EXPECT_EQ(exts.size(), 6);
  EXPECT_EQ((stencil->getStencilInstantiation().getNameFromAccessID(1)), "u");
  EXPECT_EQ((stencil->getStencilInstantiation().getNameFromAccessID(2)), "out");
  EXPECT_EQ((stencil->getStencilInstantiation().getNameFromAccessID(3)), "coeff");
  EXPECT_EQ(exts[1], (Extents{-2,2,-2,2,0,0}));
  EXPECT_EQ(exts[2], (Extents{0,0,0,0,0,0}));
  EXPECT_EQ(exts[3], (Extents{0,0,0,0,0,0}));
}
TEST_F(ComputeMaxExtents, test_stencil_03) {
  auto stencils = loadTest("compute_extent_test_stencil_03.sir");
  ASSERT_TRUE((stencils.size() == 1));
  std::shared_ptr<Stencil> stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 4));
  auto exts = stencil->computeEnclosingAccessExtents();  
  EXPECT_EQ(exts.size(), 7);
  EXPECT_EQ((stencil->getStencilInstantiation().getNameFromAccessID(1)), "u");
  EXPECT_EQ((stencil->getStencilInstantiation().getNameFromAccessID(2)), "out");
  EXPECT_EQ((stencil->getStencilInstantiation().getNameFromAccessID(3)), "coeff");
  EXPECT_EQ(exts[1], (Extents{-2,2,-2,3,0,0}));
  EXPECT_EQ(exts[2], (Extents{0,0,0,0,0,0}));
  EXPECT_EQ(exts[3], (Extents{0,0,0,1,0,0}));
}

TEST_F(ComputeMaxExtents, test_stencil_04) {
  auto stencils = loadTest("compute_extent_test_stencil_04.sir");

  ASSERT_TRUE((stencils.size() == 1));
  std::shared_ptr<Stencil> stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 4));
  auto exts = stencil->computeEnclosingAccessExtents();  
  EXPECT_EQ(exts.size(), 6);
  EXPECT_EQ((stencil->getStencilInstantiation().getNameFromAccessID(1)), "u");
  EXPECT_EQ((stencil->getStencilInstantiation().getNameFromAccessID(2)), "out");
  EXPECT_EQ(exts[1], (Extents{-3,4,-2,1,0,0}));
  EXPECT_EQ(exts[2], (Extents{0,0,0,0,0,0}));
}

TEST_F(ComputeMaxExtents, test_stencil_05) {
  auto stencils = loadTest("compute_extent_test_stencil_05.sir");
  ASSERT_TRUE((stencils.size() == 1));
  std::shared_ptr<Stencil> stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 4));
  auto exts = stencil->computeEnclosingAccessExtents();  
  EXPECT_EQ(exts.size(), 6);
  EXPECT_EQ((stencil->getStencilInstantiation().getNameFromAccessID(1)), "u");
  EXPECT_EQ((stencil->getStencilInstantiation().getNameFromAccessID(2)), "out");
  EXPECT_EQ(exts[1], (Extents{-3,4,-2,1,0,0}));
  EXPECT_EQ(exts[2], (Extents{0,0,0,0,0,0}));
}

} // anonymous namespace
