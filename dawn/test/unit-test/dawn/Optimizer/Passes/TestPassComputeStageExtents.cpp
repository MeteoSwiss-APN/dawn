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
  dawn::DawnCompiler compiler_;

protected:
  virtual void SetUp() {}

  std::unique_ptr<iir::IIR> loadTest(std::string sirFilename) {

    std::string filename = TestEnvironment::path_ + "/" + sirFilename;
    std::ifstream file(filename);
    DAWN_ASSERT_MSG((file.good()), std::string("File " + filename + " does not exists").c_str());

    std::string jsonstr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::shared_ptr<SIR> sir =
        SIRSerializer::deserializeFromString(jsonstr, SIRSerializer::Format::Json);

    auto stencilInstantiationMap = compiler_.optimize(compiler_.parallelize(sir));
    // Report diganostics
    if(compiler_.getDiagnostics().hasDiags()) {
      for(const auto& diag : compiler_.getDiagnostics().getQueue())
        std::cerr << "Compilation Error " << diag->getMessage() << std::endl;
      throw std::runtime_error("compilation failed");
    }

    DAWN_ASSERT_MSG(stencilInstantiationMap.count("compute_extent_test_stencil"),
                    "compute_extent_test_stencil not found in sir");

    std::unique_ptr<iir::IIR>& iir =
        stencilInstantiationMap["compute_extent_test_stencil"]->getIIR();
    return std::move(iir);
  }
};

TEST_F(ComputeStageExtents, test_stencil_01) {
  std::unique_ptr<iir::IIR> IIR = loadTest("compute_extent_test_stencil_01.sir");
  const auto& stencils = IIR->getChildren();

  EXPECT_EQ(stencils.size(), 1);
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getNumStages(), 2);
  EXPECT_EQ(stencil->getStage(0)->getExtents(), iir::Extents(ast::cartesian, -1, 1, -1, 1, 0, 0));
  EXPECT_EQ(stencil->getStage(1)->getExtents(), iir::Extents(ast::cartesian));
}

TEST_F(ComputeStageExtents, test_stencil_02) {
  std::unique_ptr<iir::IIR> IIR = loadTest("compute_extent_test_stencil_02.sir");
  const auto& stencils = IIR->getChildren();

  EXPECT_EQ(stencils.size(), 1);
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getNumStages(), 3);
  EXPECT_EQ(stencil->getStage(0)->getExtents(), iir::Extents(ast::cartesian, -1, 1, -1, 1, 0, 0));
  EXPECT_EQ(stencil->getStage(1)->getExtents(), iir::Extents(ast::cartesian, -1, 0, -1, 0, 0, 0));
  EXPECT_EQ(stencil->getStage(2)->getExtents(), iir::Extents(ast::cartesian));
}
TEST_F(ComputeStageExtents, test_stencil_03) {
  std::unique_ptr<iir::IIR> IIR = loadTest("compute_extent_test_stencil_03.sir");
  const auto& stencils = IIR->getChildren();
  EXPECT_EQ(stencils.size(), 1);
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getNumStages(), 4);
  EXPECT_EQ(stencil->getStage(0)->getExtents(), iir::Extents(ast::cartesian, -1, 1, -1, 2, 0, 0));
  EXPECT_EQ(stencil->getStage(1)->getExtents(), iir::Extents(ast::cartesian, -1, 0, -1, 1, 0, 0));
  EXPECT_EQ(stencil->getStage(2)->getExtents(), iir::Extents(ast::cartesian, 0, 0, 0, 1, 0, 0));
  EXPECT_EQ(stencil->getStage(3)->getExtents(), iir::Extents(ast::cartesian));
}

TEST_F(ComputeStageExtents, test_stencil_04) {
  std::unique_ptr<iir::IIR> IIR = loadTest("compute_extent_test_stencil_04.sir");
  const auto& stencils = IIR->getChildren();

  EXPECT_EQ(stencils.size(), 1);
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getNumStages(), 4);
  EXPECT_EQ(stencil->getStage(0)->getExtents(), iir::Extents(ast::cartesian, -2, 3, -2, 1, 0, 0));
  EXPECT_EQ(stencil->getStage(1)->getExtents(), iir::Extents(ast::cartesian, -1, 1, -1, 0, 0, 0));
  EXPECT_EQ(stencil->getStage(2)->getExtents(), iir::Extents(ast::cartesian, 0, 0, -1, 0, 0, 0));
  EXPECT_EQ(stencil->getStage(3)->getExtents(), iir::Extents(ast::cartesian));
}

TEST_F(ComputeStageExtents, test_stencil_05) {
  std::unique_ptr<iir::IIR> IIR = loadTest("compute_extent_test_stencil_05.sir");
  const auto& stencils = IIR->getChildren();
  ASSERT_TRUE((stencils.size() == 1));
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getNumStages(), 4);
  EXPECT_EQ(stencil->getStage(0)->getExtents(), iir::Extents(ast::cartesian, -2, 3, -2, 1, 0, 0));
  EXPECT_EQ(stencil->getStage(1)->getExtents(), iir::Extents(ast::cartesian, -1, 0, -1, 0, 0, 0));
  EXPECT_EQ(stencil->getStage(2)->getExtents(), iir::Extents(ast::cartesian, 0, 1, -1, 0, 0, 0));
  EXPECT_EQ(stencil->getStage(3)->getExtents(), iir::Extents(ast::cartesian));
}

} // anonymous namespace
