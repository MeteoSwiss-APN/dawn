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

class ComputeMaxExtents : public ::testing::Test {
  dawn::DawnCompiler compiler_;

protected:
  virtual void SetUp() {}

  const std::shared_ptr<iir::StencilInstantiation> loadTest(std::string sirFilename) {

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

    return stencilInstantiationMap["compute_extent_test_stencil"];
  }
};

TEST_F(ComputeMaxExtents, test_stencil_01) {
  const std::shared_ptr<iir::StencilInstantiation>& instantiation =
      loadTest("compute_extent_test_stencil_01.sir");
  const auto& metadata = instantiation->getMetaData();
  const std::unique_ptr<iir::IIR>& IIR = instantiation->getIIR();
  const auto& stencils = IIR->getChildren();
  ASSERT_TRUE((stencils.size() == 1));
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 2));
  auto exts = stencil->getFields();
  EXPECT_EQ(exts.size(), 3);
  int u_id = metadata.getAccessIDFromName("u");
  int out_id = metadata.getAccessIDFromName("out");
  int lap_id = metadata.getAccessIDFromName("lap");

  EXPECT_EQ(exts.at(u_id).field.getExtentsRB(),
            (iir::Extents(dawn::ast::cartesian, -2, 2, -2, 2, 0, 0)));
  EXPECT_EQ(exts.at(out_id).field.getExtentsRB(),
            (iir::Extents(dawn::ast::cartesian, 0, 0, 0, 0, 0, 0)));
  EXPECT_EQ(exts.at(lap_id).field.getExtentsRB(),
            (iir::Extents(dawn::ast::cartesian, -1, 1, -1, 1, 0, 0)));
}

TEST_F(ComputeMaxExtents, test_stencil_02) {
  const std::shared_ptr<iir::StencilInstantiation>& instantiation =
      loadTest("compute_extent_test_stencil_02.sir");
  const auto& metadata = instantiation->getMetaData();
  const std::unique_ptr<iir::IIR>& IIR = instantiation->getIIR();
  const auto& stencils = IIR->getChildren();
  ASSERT_TRUE((stencils.size() == 1));
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 3));
  auto exts = stencil->getFields();
  EXPECT_EQ(exts.size(), 6);
  int u_id = metadata.getAccessIDFromName("u");
  int out_id = metadata.getAccessIDFromName("out");
  int coeff_id = metadata.getAccessIDFromName("coeff");

  EXPECT_EQ(exts.at(u_id).field.getExtentsRB(),
            (iir::Extents(dawn::ast::cartesian, -2, 2, -2, 2, 0, 0)));
  EXPECT_EQ(exts.at(out_id).field.getExtentsRB(),
            (iir::Extents(dawn::ast::cartesian, 0, 0, 0, 0, 0, 0)));
  EXPECT_EQ(exts.at(coeff_id).field.getExtentsRB(),
            (iir::Extents(dawn::ast::cartesian, 0, 0, 0, 0, 0, 0)));
}
TEST_F(ComputeMaxExtents, test_stencil_03) {
  const std::shared_ptr<iir::StencilInstantiation>& instantiation =
      loadTest("compute_extent_test_stencil_03.sir");
  const auto& metadata = instantiation->getMetaData();
  const std::unique_ptr<iir::IIR>& IIR = instantiation->getIIR();
  const auto& stencils = IIR->getChildren();
  ASSERT_TRUE((stencils.size() == 1));
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 4));
  auto exts = stencil->getFields();
  EXPECT_EQ(exts.size(), 7);
  int u_id = metadata.getAccessIDFromName("u");
  int out_id = metadata.getAccessIDFromName("out");
  int coeff_id = metadata.getAccessIDFromName("coeff");

  EXPECT_EQ(exts.at(u_id).field.getExtentsRB(),
            (iir::Extents(dawn::ast::cartesian, -2, 2, -2, 3, 0, 0)));
  EXPECT_EQ(exts.at(out_id).field.getExtentsRB(),
            (iir::Extents(dawn::ast::cartesian, 0, 0, 0, 0, 0, 0)));
  EXPECT_EQ(exts.at(coeff_id).field.getExtentsRB(),
            (iir::Extents(dawn::ast::cartesian, 0, 0, 0, 1, 0, 0)));
}

TEST_F(ComputeMaxExtents, test_stencil_04) {
  const std::shared_ptr<iir::StencilInstantiation>& instantiation =
      loadTest("compute_extent_test_stencil_04.sir");
  const auto& metadata = instantiation->getMetaData();
  const std::unique_ptr<iir::IIR>& IIR = instantiation->getIIR();
  const auto& stencils = IIR->getChildren();

  ASSERT_TRUE((stencils.size() == 1));
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 4));
  auto exts = stencil->getFields();
  EXPECT_EQ(exts.size(), 6);

  int u_id = metadata.getAccessIDFromName("u");
  int out_id = metadata.getAccessIDFromName("out");
  EXPECT_EQ(exts.at(u_id).field.getExtentsRB(),
            (iir::Extents(dawn::ast::cartesian, -3, 4, -2, 1, 0, 0)));
  EXPECT_EQ(exts.at(out_id).field.getExtentsRB(),
            (iir::Extents(dawn::ast::cartesian, 0, 0, 0, 0, 0, 0)));
}

TEST_F(ComputeMaxExtents, test_stencil_05) {
  const std::shared_ptr<iir::StencilInstantiation>& instantiation =
      loadTest("compute_extent_test_stencil_05.sir");
  const auto& metadata = instantiation->getMetaData();
  const std::unique_ptr<iir::IIR>& IIR = instantiation->getIIR();
  const auto& stencils = IIR->getChildren();

  ASSERT_TRUE((stencils.size() == 1));
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 4));
  auto exts = stencil->getFields();
  EXPECT_EQ(exts.size(), 6);
  int u_id = metadata.getAccessIDFromName("u");
  int out_id = metadata.getAccessIDFromName("out");

  EXPECT_EQ(exts.at(u_id).field.getExtentsRB(),
            (iir::Extents(dawn::ast::cartesian, -3, 4, -2, 1, 0, 0)));
  EXPECT_EQ(exts.at(out_id).field.getExtentsRB(),
            (iir::Extents(dawn::ast::cartesian, 0, 0, 0, 0, 0, 0)));
}

} // anonymous namespace
