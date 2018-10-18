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

using namespace dawn;

namespace {

class TestFieldAccessIntervals : public ::testing::Test {
  std::unique_ptr<dawn::Options> compileOptions_;

  dawn::DawnCompiler compiler_;

protected:
  TestFieldAccessIntervals() : compiler_(compileOptions_.get()) {}
  virtual void SetUp() {}

  std::unique_ptr<iir::IIR> loadTest(std::string sirFilename) {

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

    DAWN_ASSERT_MSG((optimizer->getNameIIRMap().count("compute_extent_test_stencil")),
                    "compute_extent_test_stencil not found in sir");

    return optimizer->getNameIIRMap()["compute_extent_test_stencil"]->clone();
  }
};

TEST_F(TestFieldAccessIntervals, test_field_access_interval_01) {
  auto iir = loadTest("test_field_access_interval_01.sir");
  const auto& stencils = iir->getChildren();
  EXPECT_EQ(stencils.size(), 1);
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getNumStages(), 2);
  EXPECT_EQ(stencil->getStage(0)->getExtents(), (iir::Extents{-1, 1, -1, 1, 0, 0}));
  EXPECT_EQ(stencil->getStage(1)->getExtents(), (iir::Extents{0, 0, 0, 0, 0, 0}));

  for(auto fieldPair : (*stencil->childrenBegin())->getFields()) {
    iir::Field& field = fieldPair.second;
    int AccessID = fieldPair.first;

    if(AccessID == iir->getMetaData()->getAccessIDFromName("lap")) {
      EXPECT_EQ(field.getInterval(),
                (iir::Interval{sir::Interval::Start, sir::Interval::End, 11, 0}));
    }
    if(AccessID == iir->getMetaData()->getAccessIDFromName("out") ||
       AccessID == iir->getMetaData()->getAccessIDFromName("u")) {
      EXPECT_EQ(field.getInterval(),
                (iir::Interval{sir::Interval::Start, sir::Interval::End, 0, 0}));
    }
  }
}

TEST_F(TestFieldAccessIntervals, test_field_access_interval_02) {
  auto iir = loadTest("test_field_access_interval_02.sir");
  const auto& stencils = iir->getChildren();
  EXPECT_EQ(stencils.size(), 1);
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getNumStages(), 2);
  EXPECT_EQ(stencil->getStage(0)->getExtents(), (iir::Extents{-1, 1, -1, 1, 0, 0}));
  EXPECT_EQ(stencil->getStage(1)->getExtents(), (iir::Extents{0, 0, 0, 0, 0, 0}));

  for(auto fieldPair : (*stencil->childrenBegin())->getFields()) {
    iir::Field& field = fieldPair.second;
    int AccessID = fieldPair.first;
    if(AccessID == iir->getMetaData()->getAccessIDFromName("lap")) {
      EXPECT_EQ(field.getInterval(),
                (iir::Interval{sir::Interval::Start + 11, sir::Interval::End}));
    }
    if(AccessID == iir->getMetaData()->getAccessIDFromName("out") ||
       AccessID == iir->getMetaData()->getAccessIDFromName("u")) {
      EXPECT_EQ(field.getInterval(), (iir::Interval{sir::Interval::Start, sir::Interval::End}));
    }
    if(AccessID == iir->getMetaData()->getAccessIDFromName("coeff")) {
      EXPECT_EQ(field.getInterval(),
                (iir::Interval{sir::Interval::Start, sir::Interval::End, 11, 0}));
      EXPECT_EQ(field.computeAccessedInterval(), (iir::Interval{12, sir::Interval::End + 1}));
    }
  }
}

TEST_F(TestFieldAccessIntervals, test_field_access_interval_03) {
  auto iir = loadTest("test_field_access_interval_03.sir");
  const auto& stencils = iir->getChildren();
  EXPECT_EQ(stencils.size(), 1);
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getNumStages(), 3);
  EXPECT_EQ(stencil->getStage(0)->getExtents(), (iir::Extents{-1, 1, -1, 1, 0, 0}));
  EXPECT_EQ(stencil->getStage(1)->getExtents(), (iir::Extents{0, 0, 0, 0, 0, 0}));
  EXPECT_EQ(stencil->getStage(2)->getExtents(), (iir::Extents{0, 0, 0, 0, 0, 0}));

  for(auto fieldPair : (*stencil->childrenBegin())->getFields()) {
    iir::Field& field = fieldPair.second;
    int AccessID = fieldPair.first;
    if(AccessID == iir->getMetaData()->getAccessIDFromName("lap")) {
      EXPECT_EQ(field.getInterval(),
                (iir::Interval{sir::Interval::Start + 11, sir::Interval::End}));
    }
    if(AccessID == iir->getMetaData()->getAccessIDFromName("out") ||
       AccessID == iir->getMetaData()->getAccessIDFromName("u")) {
      EXPECT_EQ(field.getInterval(), (iir::Interval{sir::Interval::Start, sir::Interval::End}));
    }
    if(AccessID == iir->getMetaData()->getAccessIDFromName("coeff")) {
      EXPECT_EQ(field.getInterval(),
                (iir::Interval{sir::Interval::Start, sir::Interval::End, 4, 0}));
      EXPECT_EQ(field.computeAccessedInterval(),
                (iir::Interval{sir::Interval::Start, sir::Interval::End, 2, 1}));
    }
  }
}

TEST_F(TestFieldAccessIntervals, test_field_access_interval_04) {
  auto iir = loadTest("test_field_access_interval_04.sir");
  const auto& stencils = iir->getChildren();
  EXPECT_EQ(stencils.size(), 1);
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getNumStages(), 3);

  iir::MultiStage& multiStage = **(stencil->childrenBegin());

  boost::optional<iir::Interval> enclosingInterval =
      multiStage.getEnclosingAccessIntervalTemporaries();
  ASSERT_TRUE(enclosingInterval.is_initialized());
  EXPECT_EQ((*enclosingInterval), (iir::Interval{2, 14}));
}

TEST_F(TestFieldAccessIntervals, test_field_access_interval_05) {
  auto iir = loadTest("test_field_access_interval_05.sir");
  const auto& stencils = iir->getChildren();
  EXPECT_EQ(stencils.size(), 1);
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getNumStages(), 2);
  EXPECT_EQ(stencil->getStage(0)->getExtents(), (iir::Extents{-1, 1, -1, 1, 0, 0}));
  EXPECT_EQ(stencil->getStage(1)->getExtents(), (iir::Extents{0, 0, 0, 0, 0, 0}));

  for(auto fieldPair : (*stencil->childrenBegin())->getFields()) {
    iir::Field& field = fieldPair.second;
    int AccessID = fieldPair.first;
    if(AccessID == iir->getMetaData()->getAccessIDFromName("lap")) {
      EXPECT_EQ(field.getInterval(),
                (iir::Interval{sir::Interval::Start + 11, sir::Interval::End}));
      EXPECT_EQ(field.computeAccessedInterval(),
                (iir::Interval{sir::Interval::Start + 10, sir::Interval::End}));
    }
    if(AccessID == iir->getMetaData()->getAccessIDFromName("out") ||
       AccessID == iir->getMetaData()->getAccessIDFromName("u")) {
      EXPECT_EQ(field.getInterval(), (iir::Interval{sir::Interval::Start, sir::Interval::End}));
    }
    if(AccessID == iir->getMetaData()->getAccessIDFromName("coeff")) {
      EXPECT_EQ(field.getInterval(),
                (iir::Interval{sir::Interval::Start, sir::Interval::End, 11, 0}));
      EXPECT_EQ(field.computeAccessedInterval(), (iir::Interval{12, sir::Interval::End + 1}));
    }
  }
}

} // anonymous namespace
