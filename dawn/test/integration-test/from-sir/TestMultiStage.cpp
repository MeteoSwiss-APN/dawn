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

using namespace dawn;

namespace dawn {

class TestMultiStage : public TestFromSIR {};

TEST_F(TestMultiStage, test_compute_ordered_do_methods) {
  //    Stencil_0
  //    {
  //      MultiStage_0 [forward]
  //      {
  //        Stage_0
  //        {
  //          Do_0 { Start : Start }
  //          {
  //            tmp[0, 0, 0] = a[0, 0, 0];
  //              Write Accesses:
  //                tmp : [(0, 0), (0, 0), (0, 0)]
  //              Read Accesses:
  //                a : [(0, 0), (0, 0), (0, 0)]

  //          }
  //          Do_1 { Start+1 : End-1 }
  //          {
  //            tmp[0, 0, 0] = tmp[0, 0, -1];
  //              Write Accesses:
  //                tmp : [(0, 0), (0, 0), (0, 0)]
  //              Read Accesses:
  //                tmp : [(0, 0), (0, 0), (-1, 0)]

  //          }
  //          Extents: [(0, 1), (0, 0), (-1, 0)]
  //        }
  //        Stage_1
  //        {
  //          Do_0 { Start : Start }
  //          {
  //            b[0, 0, 0] = tmp[0, 0, 0];
  //              Write Accesses:
  //                b : [(0, 0), (0, 0), (0, 0)]
  //              Read Accesses:
  //                tmp : [(0, 0), (0, 0), (0, 0)]

  //          }
  //          Do_1 { Start+1 : End-4 }
  //          {
  //            b[0, 0, 0] = tmp[1, 0, -1];
  //              Write Accesses:
  //                b : [(0, 0), (0, 0), (0, 0)]
  //              Read Accesses:
  //                tmp : [(0, 1), (0, 0), (-1, 0)]

  //          }
  //          Do_2 { End : End }
  //          {
  //            tmp[0, 0, 0] = (tmp[0, 0, -1] + a[0, 0, 0]);
  //              Write Accesses:
  //                tmp : [(0, 0), (0, 0), (0, 0)]
  //              Read Accesses:
  //                a : [(0, 0), (0, 0), (0, 0)]
  //                tmp : [(0, 0), (0, 0), (-1, 0)]

  //          }
  //          Extents: [(0, 1), (0, 0), (-1, 0)]
  //        }
  //        Stage_2
  //        {
  //          Do_0 { End-3 : End }
  //          {
  //            b[0, 0, 0] = (tmp[1, 0, -1] + a[0, 0, 0]);
  //              Write Accesses:
  //                b : [(0, 0), (0, 0), (0, 0)]
  //              Read Accesses:
  //                a : [(0, 0), (0, 0), (0, 0)]
  //                tmp : [(0, 1), (0, 0), (-1, 0)]

  //          }
  //          Extents: [(0, 0), (0, 0), (0, 0)]
  //        }
  //      }
  //    }

  auto stencilInstantiation = loadTest("input/test_compute_ordered_do_methods.sir", "stencil");
  const auto& stencils = stencilInstantiation->getStencils();
  EXPECT_EQ(stencils.size(), 1);
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getChildren().size(), 1);

  auto const& mss = (*stencil->childrenBegin());
  EXPECT_EQ(mss->getChildren().size(), 3);

  auto stageit = mss->getChildren().begin();
  auto const& stage0 = *stageit;
  EXPECT_EQ(stage0->getChildren().size(), 2);

  auto const& do0_0 = stage0->getChildren().at(0);
  auto const& do0_1 = stage0->getChildren().at(1);

  stageit++;
  auto const& stage1 = *stageit;
  EXPECT_EQ(stage1->getChildren().size(), 3);
  auto const& do1_0 = stage1->getChildren().at(0);
  auto const& do1_1 = stage1->getChildren().at(1);
  auto const& do1_2 = stage1->getChildren().at(2);

  stageit++;
  auto const& stage2 = *stageit;
  EXPECT_EQ(stage2->getChildren().size(), 1);
  auto const& do2_0 = stage2->getChildren().at(0);

  auto orderedDoMethods = mss->computeOrderedDoMethods();
  EXPECT_EQ(orderedDoMethods.size(), 8);

  EXPECT_EQ(orderedDoMethods[0]->getInterval(), (iir::Interval{0, 0}));
  EXPECT_EQ(orderedDoMethods[0]->getID(), do0_0->getID());

  EXPECT_EQ(orderedDoMethods[1]->getInterval(), (iir::Interval{0, 0}));
  EXPECT_EQ(orderedDoMethods[1]->getID(), do1_0->getID());

  EXPECT_EQ(orderedDoMethods[2]->getInterval(), (iir::Interval{1, sir::Interval::End - 4}));
  EXPECT_EQ(orderedDoMethods[2]->getID(), do0_1->getID());

  EXPECT_EQ(orderedDoMethods[3]->getInterval(), (iir::Interval{1, sir::Interval::End - 4}));
  EXPECT_EQ(orderedDoMethods[3]->getID(), do1_1->getID());

  EXPECT_EQ(orderedDoMethods[4]->getInterval(),
            (iir::Interval{sir::Interval::End - 3, sir::Interval::End - 1}));
  EXPECT_EQ(orderedDoMethods[4]->getID(), do0_1->getID());

  EXPECT_EQ(orderedDoMethods[5]->getInterval(),
            (iir::Interval{sir::Interval::End - 3, sir::Interval::End - 1}));
  EXPECT_EQ(orderedDoMethods[5]->getID(), do2_0->getID());

  EXPECT_EQ(orderedDoMethods[6]->getInterval(),
            (iir::Interval{sir::Interval::End, sir::Interval::End}));
  EXPECT_EQ(orderedDoMethods[6]->getID(), do1_2->getID());

  EXPECT_EQ(orderedDoMethods[7]->getInterval(),
            (iir::Interval{sir::Interval::End, sir::Interval::End}));
  EXPECT_EQ(orderedDoMethods[7]->getID(), do2_0->getID());
}

TEST_F(TestMultiStage, test_compute_read_access_interval) {
  //    Stencil_0
  //    {
  //      MultiStage_0 [forward]
  //      {
  //        Stage_0
  //        {
  //          Do_0 { Start+2 : End-1 }
  //          {
  //            tmp[0, 0, 0] = tmp[0, 0, -2];
  //              Write Accesses:
  //                tmp : [(0, 0), (0, 0), (0, 0)]
  //              Read Accesses:
  //                tmp : [(0, 0), (0, 0), (-2, 0)]

  //          }
  //          Do_1 { End : End }
  //          {
  //            tmp[0, 0, 0] = (tmp[0, 0, -1] + a[0, 0, 0]);
  //              Write Accesses:
  //                tmp : [(0, 0), (0, 0), (0, 0)]
  //              Read Accesses:
  //                a : [(0, 0), (0, 0), (0, 0)]
  //                tmp : [(0, 0), (0, 0), (-1, 0)]

  //          }
  //          Extents: [(0, 0), (0, 0), (0, 0)]
  //        }
  //      }
  //    }

  auto stencilInstantiation = loadTest("input/test_compute_read_access_interval.sir", "stencil");
  const auto& stencils = stencilInstantiation->getStencils();
  EXPECT_EQ(stencils.size(), 1);
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getChildren().size(), 1);

  auto const& mss = *stencil->childrenBegin();

  int accessID = stencilInstantiation->getMetaData().getAccessIDFromName("tmp");
  auto interval = mss->computeReadAccessInterval(accessID);

  EXPECT_EQ(interval, (iir::MultiInterval{iir::Interval{0, 1}}));
}

TEST_F(TestMultiStage, DISABLED_test_compute_read_access_interval_02) {

  //    Stencil_0
  //    {
  //      MultiStage_0 [forward]
  //      {
  //        Stage_0
  //        {
  //          Do_0 { End-3 : End }
  //          {
  //            b[0, 0, 0] = (tmp[1, 0, 1] + a[0, 0, 0]);
  //              Write Accesses:
  //                b : [(0, 0), (0, 0), (0, 0)]
  //              Read Accesses:
  //                a : [(0, 0), (0, 0), (0, 0)]
  //                tmp : [(0, 1), (0, 0), (0, 1)]

  //          }
  //          Do_1 { Start+2 : End-4 }
  //          {
  //            tmp[0, 0, 0] = tmp[0, 0, -2];
  //              Write Accesses:
  //                tmp : [(0, 0), (0, 0), (0, 0)]
  //              Read Accesses:
  //                tmp : [(0, 0), (0, 0), (-2, 0)]

  //          }
  //          Do_2 { Start : Start }
  //          {
  //            b[0, 0, 0] = tmp[-2, 0, 0];
  //              Write Accesses:
  //                b : [(0, 0), (0, 0), (0, 0)]
  //              Read Accesses:
  //                tmp : [(-2, 0), (0, 0), (0, 0)]

  //          }
  //          Extents: [(0, 1), (0, 0), (-1, 0)]
  //        }
  //        Stage_1
  //        {
  //          Do_0 { Start+1 : End-4 }
  //          {
  //            b[0, 0, 0] = tmp[1, 0, -1];
  //              Write Accesses:
  //                b : [(0, 0), (0, 0), (0, 0)]
  //              Read Accesses:
  //                tmp : [(0, 1), (0, 0), (-1, 0)]

  //          }
  //          Extents: [(0, 0), (0, 0), (0, 0)]
  //        }
  //      }
  //    }

  auto stencilInstantiation = loadTest("input/test_compute_read_access_interval_02.sir", "stencil");
  const auto& stencils = stencilInstantiation->getStencils();
  EXPECT_EQ(stencils.size(), 1);
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getChildren().size(), 1);

  auto const& mss = *stencil->childrenBegin();

  int accessID = stencilInstantiation->getMetaData().getAccessIDFromName("tmp");
  auto interval = mss->computeReadAccessInterval(accessID);

  EXPECT_EQ(interval,
            (iir::MultiInterval{iir::Interval{0, 1},
                                iir::Interval{sir::Interval::End - 2, sir::Interval::End + 1}}));
}

TEST_F(TestMultiStage, test_field_access_interval_04) {

  //    Stencil_0
  //    {
  //      MultiStage_0 [backward]
  //      {
  //        Stage_0
  //        {
  //          Do_0 { Start : Start+8 }
  //          {
  //            out2[0, 0, 0] = u[0, 0, 6];
  //              Write Accesses:
  //                out2 : [(0, 0), (0, 0), (0, 0)]
  //              Read Accesses:
  //                u : [(0, 0), (0, 0), (6, 6)]

  //          }
  //          Extents: [(0, 0), (0, 0), (0, 0)]
  //        }
  //        Stage_1
  //        {
  //          Do_0 { Start+2 : Start+3 }
  //          {
  //            u[0, 0, 0] = in[0, 0, 0];
  //              Write Accesses:
  //                u : [(0, 0), (0, 0), (0, 0)]
  //              Read Accesses:
  //                in : [(0, 0), (0, 0), (0, 0)]

  //          }
  //          Extents: [(0, 0), (0, 0), (0, 2)]
  //        }
  //        Stage_2
  //        {
  //          Do_0 { Start+2 : Start+10 }
  //          {
  //            out1[0, 0, 0] = (u[0, 0, 2] + u[0, 0, 1]);
  //              Write Accesses:
  //                out1 : [(0, 0), (0, 0), (0, 0)]
  //              Read Accesses:
  //                u : [(0, 0), (0, 0), (0, 2)]

  //          }
  //          Extents: [(0, 0), (0, 0), (0, 0)]
  //        }
  //      }
  //    }

  auto stencilInstantiation =
      loadTest("input/test_field_access_interval_04.sir", "compute_extent_test_stencil");
  const auto& stencils = stencilInstantiation->getStencils();
  EXPECT_EQ(stencils.size(), 1);
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getChildren().size(), 2);

  auto const& mss = *(std::next(stencil->childrenBegin()));

  int accessID = stencilInstantiation->getMetaData().getAccessIDFromName("u");
  auto interval = mss->computeReadAccessInterval(accessID);

  EXPECT_EQ(interval, (iir::MultiInterval{iir::Interval{3, 14}}));
}

TEST_F(TestMultiStage, test_compute_read_access_interval_03) {
  /*
    vertical_region(k_start, k_start) {
      tmp = a;
    }
    vertical_region(k_start + 1, k_end) {
      b = tmp(k - 1);
    }
    vertical_region(k_end, k_end) {
      tmp = (b(k - 1) + b) * tmp;
    }
    vertical_region(k_end - 1, k_start) {
      tmp = 2 * b;
      c = tmp(k + 1);
    }
   */
  auto stencilInstantiation = loadTest("input/test_compute_read_access_interval_03.sir", "stencil");
  const auto& stencils = stencilInstantiation->getStencils();
  EXPECT_EQ(stencils.size(), 1);
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getChildren().size(), 3);

  auto const mss0it = stencil->childrenBegin();
  auto const& mss0 = *mss0it;

  int accessID = stencilInstantiation->getMetaData().getAccessIDFromName("tmp");
  auto interval0 = mss0->computeReadAccessInterval(accessID);

  EXPECT_EQ(interval0, (iir::MultiInterval{iir::Interval{1, sir::Interval::End - 1}}));

  auto const& mss1 = *(std::next(mss0it));
  auto interval1 = mss1->computeReadAccessInterval(accessID);

  EXPECT_EQ(interval1, (iir::MultiInterval{iir::Interval{sir::Interval::End, sir::Interval::End}}));
}

TEST_F(TestMultiStage, test_compute_read_access_interval_04) {
  /**
   vertical_region(k_start, k_end) {
      tmp = in;

      b1 = a1;
      c1 = b1(k + 1);
      c1 = b1(k - 1);

      out = tmp;
      tmp = in;

      b2 = a2;
      c2 = b2(k + 1);
      c2 = b2(k - 1);

      out = tmp;
    }
   */
  auto instantiation = loadTest("input/test_compute_read_access_interval_04.sir", "stencil");
  const auto& stencils = instantiation->getStencils();

  EXPECT_EQ(stencils.size(), 1);
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getChildren().size(), 1);
  auto const& mss = *(stencil->childrenBegin());
  const auto& metadata = instantiation->getMetaData();

  auto a2_interval = mss->computeReadAccessInterval(metadata.getAccessIDFromName("a2"));
  auto multinterval = iir::MultiInterval{iir::Interval{0, sir::Interval::End, 0, 0}};
  ASSERT_EQ(a2_interval, multinterval);

  auto b1_interval = mss->computeReadAccessInterval(metadata.getAccessIDFromName("b1"));
  multinterval =
      iir::MultiInterval{iir::Interval{0, 0, -1, -1}, iir::Interval{0, sir::Interval::End, 1, 1}};
  ASSERT_EQ(b1_interval, multinterval);
}
} // namespace dawn
