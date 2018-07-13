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
#include <string>

using namespace dawn;

namespace {

class MultiStageTest : public ::testing::Test {
  std::unique_ptr<dawn::Options> compileOptions_;

  dawn::DawnCompiler compiler_;

protected:
  MultiStageTest() : compiler_(compileOptions_.get()) {}
  virtual void SetUp() {}

  std::shared_ptr<StencilInstantiation> loadTest(std::string sirFilename, std::string stencilName) {

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

    DAWN_ASSERT_MSG((optimizer->getStencilInstantiationMap().count(stencilName)),
                    "compute_extent_test_stencil not found in sir");

    return optimizer->getStencilInstantiationMap()[stencilName];
  }
};

TEST_F(MultiStageTest, test_compute_ordered_do_methods) {
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

  auto stencilInstantiation = loadTest("test_compute_ordered_do_methods.sir", "stencil");
  auto stencils = stencilInstantiation->getStencils();
  EXPECT_EQ(stencils.size(), 1);
  std::shared_ptr<Stencil> stencil = stencils[0];

  EXPECT_EQ(stencil->getMultiStages().size(), 1);

  auto const& mss = stencil->getMultiStages().front();

  EXPECT_EQ(mss->getStages().size(), 3);
  auto stageit = mss->getStages().begin();
  auto const& stage0 = *stageit;
  EXPECT_EQ(stage0->getDoMethods().size(), 2);
  auto const& do0_0 = stage0->getDoMethods()[0];
  auto const& do0_1 = stage0->getDoMethods()[1];

  stageit++;
  auto const& stage1 = *stageit;
  EXPECT_EQ(stage1->getDoMethods().size(), 3);
  auto const& do1_0 = stage1->getDoMethods()[0];
  auto const& do1_1 = stage1->getDoMethods()[1];
  auto const& do1_2 = stage1->getDoMethods()[2];

  stageit++;
  auto const& stage2 = *stageit;
  EXPECT_EQ(stage2->getDoMethods().size(), 1);
  auto const& do2_0 = stage2->getDoMethods()[0];

  auto orderedDoMethods = mss->computeOrderedDoMethods();
  EXPECT_EQ(orderedDoMethods.size(), 8);

  EXPECT_EQ(orderedDoMethods[0].getInterval(), (Interval{0, 0}));
  EXPECT_EQ(orderedDoMethods[0].getID(), do0_0->getID());

  EXPECT_EQ(orderedDoMethods[1].getInterval(), (Interval{0, 0}));
  EXPECT_EQ(orderedDoMethods[1].getID(), do1_0->getID());

  EXPECT_EQ(orderedDoMethods[2].getInterval(), (Interval{1, sir::Interval::End - 4}));
  EXPECT_EQ(orderedDoMethods[2].getID(), do0_1->getID());

  EXPECT_EQ(orderedDoMethods[3].getInterval(), (Interval{1, sir::Interval::End - 4}));
  EXPECT_EQ(orderedDoMethods[3].getID(), do1_1->getID());

  EXPECT_EQ(orderedDoMethods[4].getInterval(),
            (Interval{sir::Interval::End - 3, sir::Interval::End - 1}));
  EXPECT_EQ(orderedDoMethods[4].getID(), do0_1->getID());

  EXPECT_EQ(orderedDoMethods[5].getInterval(),
            (Interval{sir::Interval::End - 3, sir::Interval::End - 1}));
  EXPECT_EQ(orderedDoMethods[5].getID(), do2_0->getID());

  EXPECT_EQ(orderedDoMethods[6].getInterval(), (Interval{sir::Interval::End, sir::Interval::End}));
  EXPECT_EQ(orderedDoMethods[6].getID(), do1_2->getID());

  EXPECT_EQ(orderedDoMethods[7].getInterval(), (Interval{sir::Interval::End, sir::Interval::End}));
  EXPECT_EQ(orderedDoMethods[7].getID(), do2_0->getID());
}

TEST_F(MultiStageTest, test_compute_read_access_interval) {

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

  auto stencilInstantiation = loadTest("test_compute_read_access_interval.sir", "stencil");
  auto stencils = stencilInstantiation->getStencils();
  EXPECT_EQ(stencils.size(), 1);
  std::shared_ptr<Stencil> stencil = stencils[0];

  EXPECT_EQ(stencil->getMultiStages().size(), 1);

  auto const& mss = stencil->getMultiStages().front();

  int accessID = stencilInstantiation->getAccessIDFromName("tmp");
  auto interval = mss->computeReadAccessInterval(accessID);

  EXPECT_EQ(interval, (MultiInterval{Interval{0, 1}}));
}

TEST_F(MultiStageTest, test_compute_read_access_interval_02) {

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

  auto stencilInstantiation = loadTest("test_compute_read_access_interval_02.sir", "stencil");
  auto stencils = stencilInstantiation->getStencils();
  EXPECT_EQ(stencils.size(), 1);
  std::shared_ptr<Stencil> stencil = stencils[0];

  EXPECT_EQ(stencil->getMultiStages().size(), 1);

  auto const& mss = stencil->getMultiStages().front();

  int accessID = stencilInstantiation->getAccessIDFromName("tmp");
  auto interval = mss->computeReadAccessInterval(accessID);

  EXPECT_EQ(interval, (MultiInterval{Interval{0, 1},
                                     Interval{sir::Interval::End - 2, sir::Interval::End + 1}}));
}

TEST_F(MultiStageTest, test_field_access_interval_04) {

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
      loadTest("test_field_access_interval_04.sir", "compute_extent_test_stencil");
  auto stencils = stencilInstantiation->getStencils();
  EXPECT_EQ(stencils.size(), 1);
  std::shared_ptr<Stencil> stencil = stencils[0];

  EXPECT_EQ(stencil->getMultiStages().size(), 1);

  auto const& mss = stencil->getMultiStages().front();

  int accessID = stencilInstantiation->getAccessIDFromName("u");
  auto interval = mss->computeReadAccessInterval(accessID);

  EXPECT_EQ(interval, (MultiInterval{Interval{4, 14}}));
}

TEST_F(MultiStageTest, test_compute_read_access_interval_03) {
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
  //          Do_1 { Start+1 : End }
  //          {
  //            b[0, 0, 0] = tmp[0, 0, -1];
  //              Write Accesses:
  //                b : [(0, 0), (0, 0), (0, 0)]
  //              Read Accesses:
  //                tmp : [(0, 0), (0, 0), (-1, 0)]

  //          }
  //          Extents: [(0, 0), (0, 0), (-1, 0)]
  //        }
  //      }
  //      MultiStage_1 [backward]
  //      {
  //        Stage_0
  //        {
  //          Do_0 { End : End }
  //          {
  //            tmp[0, 0, 0] = ((b[0, 0, -1] + b[0, 0, 0]) * tmp[0, 0, 0]);
  //              Write Accesses:
  //                tmp : [(0, 0), (0, 0), (0, 0)]
  //              Read Accesses:
  //                tmp : [(0, 0), (0, 0), (0, 0)]
  //                b : [(0, 0), (0, 0), (-1, 0)]

  //          }
  //          Do_1 { Start : End-1 }
  //          {
  //            tmp[0, 0, 0] = (2 * b[0, 0, 0]);
  //              Write Accesses:
  //                tmp : [(0, 0), (0, 0), (0, 0)]
  //              Read Accesses:
  //                b : [(0, 0), (0, 0), (0, 0)]
  //                2 : [(0, 0), (0, 0), (0, 0)]

  //            c[0, 0, 0] = tmp[0, 0, 1];
  //              Write Accesses:
  //                c : [(0, 0), (0, 0), (0, 0)]
  //              Read Accesses:
  //                tmp : [(0, 0), (0, 0), (0, 1)]

  //          }
  //          Extents: [(0, 0), (0, 0), (0, 0)]
  //        }
  //      }
  //    }

  auto stencilInstantiation = loadTest("test_compute_read_access_interval_03.sir", "stencil");
  auto stencils = stencilInstantiation->getStencils();
  EXPECT_EQ(stencils.size(), 1);
  std::shared_ptr<Stencil> stencil = stencils[0];

  EXPECT_EQ(stencil->getMultiStages().size(), 2);

  auto const mss0it = stencil->getMultiStages().begin();
  auto const& mss0 = *mss0it;

  int accessID = stencilInstantiation->getAccessIDFromName("tmp");
  auto interval0 = mss0->computeReadAccessInterval(accessID);

  EXPECT_EQ(interval0, (MultiInterval{Interval{1, sir::Interval::End - 1}}));

  auto const& mss1 = *(std::next(mss0it));
  auto interval1 = mss1->computeReadAccessInterval(accessID);

  EXPECT_EQ(interval1, (MultiInterval{Interval{sir::Interval::End, sir::Interval::End}}));
}
TEST_F(MultiStageTest, test_compute_read_access_interval_04) {

  // Stencil_0
  //{
  //  MultiStage_0 [parallel]
  //  {
  //    Stage_0
  //    {
  //      Do_0 { Start : End }
  //      {
  //        tmp[0, 0, 0] = in[0, 0, 0];
  //          Write Accesses:
  //            tmp : [(0, 0), (0, 0), (0, 0)]
  //          Read Accesses:
  //            in : [(0, 0), (0, 0), (0, 0)]

  //        b1[0, 0, 0] = a1[0, 0, 0];
  //          Write Accesses:
  //            b1 : [(0, 0), (0, 0), (0, 0)]
  //          Read Accesses:
  //            a1 : [(0, 0), (0, 0), (0, 0)]

  //      }
  //      Extents: [(0, 0), (0, 0), (-2, 2)]
  //    }
  //  }
  //  MultiStage_1 [parallel]
  //  {
  //    Stage_0
  //    {
  //      Do_0 { Start : End }
  //      {
  //        c1[0, 0, 0] = b1[0, 0, 1];
  //          Write Accesses:
  //            c1 : [(0, 0), (0, 0), (0, 0)]
  //          Read Accesses:
  //            b1 : [(0, 0), (0, 0), (0, 1)]

  //        c1[0, 0, 0] = b1[0, 0, -1];
  //          Write Accesses:
  //            c1 : [(0, 0), (0, 0), (0, 0)]
  //          Read Accesses:
  //            b1 : [(0, 0), (0, 0), (-1, 0)]

  //        out[0, 0, 0] = tmp[0, 0, 0];
  //          Write Accesses:
  //            out : [(0, 0), (0, 0), (0, 0)]
  //          Read Accesses:
  //            tmp : [(0, 0), (0, 0), (0, 0)]

  //        tmp[0, 0, 0] = in[0, 0, 0];
  //          Write Accesses:
  //            tmp : [(0, 0), (0, 0), (0, 0)]
  //          Read Accesses:
  //            in : [(0, 0), (0, 0), (0, 0)]

  //        b2[0, 0, 0] = a2[0, 0, 0];
  //          Write Accesses:
  //            b2 : [(0, 0), (0, 0), (0, 0)]
  //          Read Accesses:
  //            a2 : [(0, 0), (0, 0), (0, 0)]

  //      }
  //      Extents: [(0, 0), (0, 0), (-1, 1)]
  //    }
  //  }
  //  MultiStage_2 [parallel]
  //  {
  //    Stage_0
  //    {
  //      Do_0 { Start : End }
  //      {
  //        c2[0, 0, 0] = b2[0, 0, 1];
  //          Write Accesses:
  //            c2 : [(0, 0), (0, 0), (0, 0)]
  //          Read Accesses:
  //            b2 : [(0, 0), (0, 0), (0, 1)]

  //        c2[0, 0, 0] = b2[0, 0, -1];
  //          Write Accesses:
  //            c2 : [(0, 0), (0, 0), (0, 0)]
  //          Read Accesses:
  //            b2 : [(0, 0), (0, 0), (-1, 0)]

  //        out[0, 0, 0] = tmp[0, 0, 0];
  //          Write Accesses:
  //            out : [(0, 0), (0, 0), (0, 0)]
  //          Read Accesses:
  //            tmp : [(0, 0), (0, 0), (0, 0)]

  //      }
  //      Extents: [(0, 0), (0, 0), (0, 0)]
  //    }
  //  }
  //}

  auto stencilInstantiation = loadTest("test_compute_read_access_interval_04.sir", "stencil");
  auto stencils = stencilInstantiation->getStencils();
  EXPECT_EQ(stencils.size(), 1);
  std::shared_ptr<Stencil> stencil = stencils[0];

  EXPECT_EQ(stencil->getMultiStages().size(), 3);

  auto const mss0it = stencil->getMultiStages().begin();
  auto const& mss0 = *(mss0it);

  int accessID = stencilInstantiation->getAccessIDFromName("tmp");
  auto interval0 = mss0->computeReadAccessInterval(accessID);

  auto const mss1it = std::next(mss0it);
  auto const& mss1 = *(mss1it);

  auto interval1 = mss1->computeReadAccessInterval(accessID);

  EXPECT_EQ(interval1, (MultiInterval{Interval{0, sir::Interval::End}}));

  auto const mss2it = std::next(mss1it);
  auto const& mss2 = *(mss2it);

  auto interval2 = mss2->computeReadAccessInterval(accessID);

  EXPECT_EQ(interval2, (MultiInterval{Interval{0, sir::Interval::End}}));
}
} // anonymous namespace
