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

#include "TestFromSIR.h"

namespace dawn {

class TestComputeEnclosingAccessInterval : public TestFromSIR {};

TEST_F(TestComputeEnclosingAccessInterval, test_field_access_interval_01) {
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
  ASSERT_TRUE((mss->getChildren().size() == 2));

  auto stage_ptr = mss->childrenBegin();
  std::unique_ptr<iir::Stage> const& stage1 = *stage_ptr;
  stage_ptr = std::next(stage_ptr);
  std::unique_ptr<iir::Stage> const& stage2 = *stage_ptr;

  std::optional<iir::Interval> intervalU1 =
      stage1->computeEnclosingAccessInterval(metadata.getAccessIDFromName("u"), false);
  ASSERT_TRUE(intervalU1.has_value());     // [k_start .. k_end]
  ASSERT_EQ(*intervalU1, (iir::Interval{0, sir::Interval::End, 0, 0}));

  std::optional<iir::Interval> intervalOut1 =
      stage1->computeEnclosingAccessInterval(metadata.getAccessIDFromName("out"), false);
  ASSERT_TRUE(intervalOut1.has_value());   // [k_start .. k_start+10]
  ASSERT_EQ(*intervalOut1, (iir::Interval{0, 0, 0, 10}));

  std::optional<iir::Interval> intervalLap1 =
      stage1->computeEnclosingAccessInterval(metadata.getAccessIDFromName("lap"), false);
  ASSERT_TRUE(intervalLap1.has_value());   // [k_start + 11 .. k_end]
  ASSERT_EQ(*intervalLap1, (iir::Interval{0, sir::Interval::End, 11, 0}));

  std::optional<iir::Interval> intervalU2 =
      stage2->computeEnclosingAccessInterval(metadata.getAccessIDFromName("u"), false);
  ASSERT_FALSE(intervalU2.has_value());

  std::optional<iir::Interval> intervalOut2 =
      stage2->computeEnclosingAccessInterval(metadata.getAccessIDFromName("out"), false);
  ASSERT_TRUE(intervalOut2.has_value());
  ASSERT_TRUE((*intervalOut2 == iir::Interval{0, sir::Interval::End, 11, 0}));

  std::optional<iir::Interval> intervalLap2 =
      stage2->computeEnclosingAccessInterval(metadata.getAccessIDFromName("lap"), false);
  ASSERT_TRUE(intervalLap2.has_value());
  ASSERT_TRUE((*intervalLap2 == iir::Interval{0, sir::Interval::End, 11, 0}));
}

TEST_F(TestComputeEnclosingAccessInterval, test_field_access_interval_02) {
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
