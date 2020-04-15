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

class TestComputeMaximumExtent : public TestFromSIR {};

TEST_F(TestComputeMaximumExtent, test_compute_maximum_extent_01) {
  auto instantiation = loadTest("input/test_compute_maximum_extent_01.sir");
  const auto& stencils = instantiation->getStencils();

  ASSERT_TRUE((stencils.size() == 1));
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];
  ASSERT_TRUE((stencil->getNumStages() == 1));
  ASSERT_TRUE((stencil->getChildren().size() == 1));

  auto const& mss = (*stencil->childrenBegin());
  auto stage1_ptr = mss->childrenBegin();
  std::unique_ptr<iir::Stage> const& stage1 = *stage1_ptr;
  ASSERT_TRUE((stage1->getChildren().size() == 1));

  const auto& doMethod1 = stage1->getSingleDoMethod();
  ASSERT_TRUE((doMethod1.computeMaximumExtents(instantiation->getMetaData().getAccessIDFromName(
                   "u")) == iir::Extents(dawn::ast::cartesian, 0, 1, -1, 0, 0, 2)));
}

TEST_F(TestComputeMaximumExtent, test_field_access_interval_02) {
  auto instantiation = loadTest("input/test_field_access_interval_02.sir");
  const auto& metadata = instantiation->getMetaData();
  const auto& stencils = instantiation->getStencils();

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
  auto extents = iir::computeMaximumExtents(*stmt, metadata.getAccessIDFromName("u"));
  ASSERT_TRUE((iir::computeMaximumExtents(*stmt, metadata.getAccessIDFromName("u")) ==
               iir::Extents(dawn::ast::cartesian, -1, 1, -1, 1, 0, 0)));

  ASSERT_EQ(iir::computeMaximumExtents(*stmt, metadata.getAccessIDFromName("coeff")),
            (iir::Extents(dawn::ast::cartesian, 0, 0, 0, 0, 1, 1)));
}

} // namespace dawn
