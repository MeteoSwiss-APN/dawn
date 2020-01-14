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
#include "dawn/IIR/ASTStmt.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"
#include <fstream>
#include <gtest/gtest.h>
#include <streambuf>
#include <string>

using namespace dawn;

namespace {

class TestComputeMaximumExtent : public ::testing::Test {
  dawn::DawnCompiler compiler_;

protected:
  virtual void SetUp() {}

  std::shared_ptr<iir::StencilInstantiation> loadTest(std::string sirFilename) {

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

TEST_F(TestComputeMaximumExtent, test_field_access_interval_02) {
  auto stencilInstantiation = loadTest("test_field_access_interval_02.sir");
  const auto& metadata = stencilInstantiation->getMetaData();
  const auto& stencils = stencilInstantiation->getStencils();
  ASSERT_TRUE((stencils.size() == 1));
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 2));
  ASSERT_TRUE((stencil->getStage(0)->getExtents() ==
               iir::Extents(dawn::ast::cartesian, -1, 1, -1, 1, 0, 0)));
  ASSERT_TRUE(
      (stencil->getStage(1)->getExtents() == iir::Extents(dawn::ast::cartesian, 0, 0, 0, 0, 0, 0)));

  ASSERT_TRUE((stencil->getChildren().size() == 1));

  auto const& mss = (*stencil->childrenBegin());

  auto stage1_ptr = mss->childrenBegin();
  std::unique_ptr<iir::Stage> const& stage1 = *stage1_ptr;

  ASSERT_TRUE((stage1->getChildren().size() == 2));

  const auto& doMethod1 = stage1->getChildren().at(0);

  ASSERT_TRUE((doMethod1->getAST().getStatements().size() == 1));
  const auto& stmt = doMethod1->getAST().getStatements()[0];
  ASSERT_TRUE((iir::computeMaximumExtents(*stmt, metadata.getAccessIDFromName("u")) ==
               iir::Extents(dawn::ast::cartesian, -1, 1, -1, 1, 0, 0)));

  EXPECT_EQ(iir::computeMaximumExtents(*stmt, metadata.getAccessIDFromName("coeff")),
            (iir::Extents(dawn::ast::cartesian, 0, 0, 0, 0, 1, 1)));
}

TEST_F(TestComputeMaximumExtent, test_compute_maximum_extent_01) {
  auto stencilInstantiation = loadTest("test_compute_maximum_extent_01.sir");
  const auto& stencils = stencilInstantiation->getStencils();

  ASSERT_TRUE((stencils.size() == 1));
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 1));
  ASSERT_TRUE((stencil->getChildren().size() == 1));

  auto const& mss = (*stencil->childrenBegin());

  auto stage1_ptr = mss->childrenBegin();
  std::unique_ptr<iir::Stage> const& stage1 = *stage1_ptr;

  ASSERT_TRUE((stage1->getChildren().size() == 1));

  const auto& doMethod1 = stage1->getSingleDoMethod();

  ASSERT_TRUE((doMethod1.computeMaximumExtents(
                   stencilInstantiation->getMetaData().getAccessIDFromName("u")) ==
               iir::Extents(dawn::ast::cartesian, 0, 1, -1, 0, 0, 2)));
}

} // anonymous namespace
