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
#include "dawn/Optimizer/PassComputeStageExtents.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "dawn/Unittest/CompilerUtil.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"
#include <fstream>
#include <gtest/gtest.h>
#include <streambuf>

using namespace dawn;

namespace {

TEST(ComputeMaxExtents, test_stencil_01) {
  /*
  vertical_region(k_start, k_end) { out = in[i - 1]; }
  */
  auto instantiation = IIRSerializer::deserialize("input/compute_extent_test_stencil_01.iir");
  const std::unique_ptr<iir::IIR>& IIR = instantiation->getIIR();
  const auto& metadata = instantiation->getMetaData();
  const auto& stencils = IIR->getChildren();
  auto exts = stencils[0]->getFields();
  int inID = metadata.getAccessIDFromName("in");
  EXPECT_EQ(exts.at(inID).field.getExtentsRB(),
            (iir::Extents(dawn::ast::cartesian, -1, -1, 0, 0, 0, 0)));
}
TEST(ComputeMaxExtents, test_stencil_02) {
  /*
  vertical_region(k_start, k_end) {
      mid = in[i - 1];
      out = in[j + 1];
    }
  */
  auto instantiation = IIRSerializer::deserialize("input/compute_extent_test_stencil_02.iir");
  const auto& metadata = instantiation->getMetaData();
  const std::unique_ptr<iir::IIR>& IIR = instantiation->getIIR();
  const auto& stencils = IIR->getChildren();
  ASSERT_TRUE((stencils.size() == 1));
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];
  auto exts = stencil->getFields();
  int inID = metadata.getAccessIDFromName("in");

  EXPECT_EQ(exts.at(inID).field.getExtentsRB(),
            (iir::Extents(dawn::ast::cartesian, -1, 0, 0, 1, 0, 0)));
}

TEST(ComputeMaxExtents, test_stencil_03) {
  /*
  vertical_region(k_start, k_end) {
      mid = in[i - 1];
      out = mid[i - 1];
    }
  */
  auto instantiation = IIRSerializer::deserialize("input/compute_extent_test_stencil_03.iir");
  const auto& metadata = instantiation->getMetaData();
  const std::unique_ptr<iir::IIR>& IIR = instantiation->getIIR();
  const auto& stencils = IIR->getChildren();
  ASSERT_TRUE((stencils.size() == 1));
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  std::unique_ptr<OptimizerContext> context;
  PassComputeStageExtents pass(*context);
  pass.run(instantiation);

  auto exts = stencil->getFields();
  int inID = metadata.getAccessIDFromName("in");

  EXPECT_EQ(exts.at(inID).field.getExtentsRB(),
            (iir::Extents(dawn::ast::cartesian, -2, -1, 0, 0, 0, 0)));
}

} // anonymous namespace
