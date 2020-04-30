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

#include "dawn/Compiler/Driver.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "dawn/Support/FileSystem.h"
#include <fstream>
#include <gtest/gtest.h>
#include <optional>
#include <streambuf>
#include <string>

using namespace dawn;

namespace {

std::shared_ptr<iir::StencilInstantiation> loadTest(const std::string& sirFilename) {
  auto sir = SIRSerializer::deserialize(sirFilename, SIRSerializer::Format::Json);
  auto stencilInstantiationMap = run(sir, {PassGroup::StageReordering, PassGroup::StageMerger});

  DAWN_ASSERT_MSG(stencilInstantiationMap.count("compute_extent_test_stencil"),
                  "compute_extent_test_stencil not found in sir");

  return stencilInstantiationMap["compute_extent_test_stencil"];
}

TEST(ComputeEnclosingAccessInterval, test_field_access_interval_01) {
  auto stencilInstantiation = loadTest("input/test_field_access_interval_01.sir");
  const auto& stencils = stencilInstantiation->getStencils();
  const auto& metadata = stencilInstantiation->getMetaData();

  ASSERT_TRUE((stencils.size() == 1));
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 2));
  ASSERT_TRUE((stencil->getStage(0)->getExtents() ==
               iir::Extents(dawn::ast::cartesian, -1, 1, -1, 1, 0, 0)));
  ASSERT_TRUE(
      (stencil->getStage(1)->getExtents() == iir::Extents(dawn::ast::cartesian, 0, 0, 0, 0, 0, 0)));

  ASSERT_TRUE((stencil->getChildren().size() == 1));

  auto const& mss = *stencil->childrenBegin();

  auto stage1_ptr = mss->childrenBegin();
  auto stage2_ptr = std::next(stage1_ptr);
  std::unique_ptr<iir::Stage> const& stage1 = *stage1_ptr;
  std::unique_ptr<iir::Stage> const& stage2 = *stage2_ptr;

  std::optional<iir::Interval> intervalU1 =
      stage1->computeEnclosingAccessInterval(metadata.getAccessIDFromName("u"), false);
  std::optional<iir::Interval> intervalOut1 =
      stage1->computeEnclosingAccessInterval(metadata.getAccessIDFromName("out"), false);
  std::optional<iir::Interval> intervalLap1 =
      stage1->computeEnclosingAccessInterval(metadata.getAccessIDFromName("lap"), false);

  ASSERT_TRUE(intervalU1.has_value());
  ASSERT_TRUE(!intervalOut1.has_value());
  ASSERT_TRUE(intervalLap1.has_value());

  ASSERT_TRUE((*intervalU1 == iir::Interval{0, sir::Interval::End, 11, 0}));
  ASSERT_TRUE((*intervalLap1 == iir::Interval{0, sir::Interval::End, 11, 0}));

  std::optional<iir::Interval> intervalU2 =
      stage2->computeEnclosingAccessInterval(metadata.getAccessIDFromName("u"), false);
  std::optional<iir::Interval> intervalOut2 =
      stage2->computeEnclosingAccessInterval(metadata.getAccessIDFromName("out"), false);
  std::optional<iir::Interval> intervalLap2 =
      stage2->computeEnclosingAccessInterval(metadata.getAccessIDFromName("lap"), false);

  ASSERT_TRUE(intervalU2.has_value());
  ASSERT_TRUE(intervalOut2.has_value());
  ASSERT_TRUE(intervalLap2.has_value());

  EXPECT_EQ(*intervalU2, (iir::Interval{0, 0, 0, 10}));
  EXPECT_EQ(*intervalOut2, (iir::Interval{0, sir::Interval::End, 0, 0}));
  EXPECT_EQ(*intervalLap2, (iir::Interval{0, sir::Interval::End, 11, 0}));
}

TEST(ComputeEnclosingAccessInterval, test_field_access_interval_02) {
  auto stencilInstantiation = loadTest("input/test_field_access_interval_02.sir");
  const auto& metadata = stencilInstantiation->getMetaData();
  const auto& stencils = stencilInstantiation->getStencils();

  ASSERT_TRUE((stencils.size() == 1));
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 2));

  ASSERT_TRUE((stencil->getChildren().size() == 1));

  auto const& mss = *stencil->childrenBegin();

  auto stage1_ptr = mss->childrenBegin();
  std::unique_ptr<iir::Stage> const& stage1 = *stage1_ptr;

  {
    std::optional<iir::Interval> intervalcoeff1 =
        stage1->computeEnclosingAccessInterval(metadata.getAccessIDFromName("coeff"), false);

    ASSERT_TRUE(intervalcoeff1.has_value());

    EXPECT_EQ(*intervalcoeff1, (iir::Interval{12, sir::Interval::End + 1}));
  }
  {
    std::optional<iir::Interval> intervalcoeff1 =
        stage1->computeEnclosingAccessInterval(metadata.getAccessIDFromName("coeff"), true);

    ASSERT_TRUE(intervalcoeff1.has_value());

    EXPECT_EQ(*intervalcoeff1, (iir::Interval{11, sir::Interval::End + 1}));
  }
}

} // anonymous namespace
