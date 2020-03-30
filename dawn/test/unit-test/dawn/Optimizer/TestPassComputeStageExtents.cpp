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

#include "dawn/IIR/IIR.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassComputeStageExtents.h"
#include "dawn/Serialization/IIRSerializer.h"
#include <fstream>
#include <gtest/gtest.h>
#include <streambuf>

using namespace dawn;

namespace {

TEST(TestComputeStageExtents, test_stencil_01) {
  /*
  vertical_region(k_start, k_end) { out = in[i - 1]; }
  */
  auto instantiation = IIRSerializer::deserialize("input/compute_extent_test_stencil_01.iir");
  const std::unique_ptr<iir::IIR>& IIR = instantiation->getIIR();
  const auto& metadata = instantiation->getMetaData();
  const auto& stencils = IIR->getChildren();
  ASSERT_TRUE((stencils.size() == 1));
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  auto fields = stencil->getFields();
  int inID = metadata.getAccessIDFromName("in");
  EXPECT_EQ(fields.at(inID).field.getExtentsRB(),
            (iir::Extents(dawn::ast::cartesian, -1, -1, 0, 0, 0, 0)));

  std::unique_ptr<OptimizerContext> context;
  PassComputeStageExtents pass(*context);
  pass.run(instantiation);

  EXPECT_EQ(stencil->getNumStages(), 1);
  EXPECT_EQ(stencil->getStage(0)->getExtents(), iir::Extents(ast::cartesian));
}

TEST(TestComputeStageExtents, test_stencil_02) {
  /*
  vertical_region(k_start, k_end) {
      mid = in[i - 1];
      out = in[j + 1];
    } */
  auto instantiation = IIRSerializer::deserialize("input/compute_extent_test_stencil_02.iir");
  const auto& metadata = instantiation->getMetaData();
  const std::unique_ptr<iir::IIR>& IIR = instantiation->getIIR();
  const auto& stencils = IIR->getChildren();
  ASSERT_TRUE((stencils.size() == 1));
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];
  auto fields = stencil->getFields();
  int inID = metadata.getAccessIDFromName("in");

  EXPECT_EQ(fields.at(inID).field.getExtentsRB(),
            (iir::Extents(dawn::ast::cartesian, -1, 0, 0, 1, 0, 0)));
}

TEST(TestComputeStageExtents, test_stencil_03) {
  /*
    vertical_region(k_start, k_end) {
      mid = in[i - 1];
      out = mid[i - 1];
    } */
  auto instantiation = IIRSerializer::deserialize("input/compute_extent_test_stencil_03.iir");
  const auto& metadata = instantiation->getMetaData();
  const std::unique_ptr<iir::IIR>& IIR = instantiation->getIIR();
  const auto& stencils = IIR->getChildren();
  ASSERT_TRUE((stencils.size() == 1));
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  std::unique_ptr<OptimizerContext> context;
  PassComputeStageExtents pass(*context);
  pass.run(instantiation);

  EXPECT_EQ(stencil->getNumStages(), 2);
  EXPECT_EQ(stencil->getStage(0)->getExtents(), iir::Extents(ast::cartesian, -1, 0, 0, 0, 0, 0));
  EXPECT_EQ(stencil->getStage(1)->getExtents(), iir::Extents(ast::cartesian));

  auto fields = stencil->getFields();
  int inID = metadata.getAccessIDFromName("in");

  EXPECT_EQ(fields.at(inID).field.getExtentsRB(),
            (iir::Extents(dawn::ast::cartesian, -2, -1, 0, 0, 0, 0)));
}

TEST(TestComputeStageExtents, test_stencil_04) {
  /*
    vertical_region(k_start, k_end) {
      mid = in;
      mid2 = mid[i + 1];
      out = mid2[j - 1];
    } */
  auto instantiation = IIRSerializer::deserialize("input/compute_extent_test_stencil_04.iir");
  const auto& stencils = instantiation->getIIR()->getChildren();
  ASSERT_TRUE((stencils.size() == 1));
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  std::unique_ptr<OptimizerContext> context;
  PassComputeStageExtents pass(*context);
  pass.run(instantiation);

  EXPECT_EQ(stencil->getNumStages(), 3);
  EXPECT_EQ(stencil->getStage(0)->getExtents(), iir::Extents(ast::cartesian, 0, 1, -1, 0, 0, 0));
  EXPECT_EQ(stencil->getStage(1)->getExtents(), iir::Extents(ast::cartesian, 0, 0, -1, 0, 0, 0));
  EXPECT_EQ(stencil->getStage(2)->getExtents(), iir::Extents(ast::cartesian));
}

} // anonymous namespace
