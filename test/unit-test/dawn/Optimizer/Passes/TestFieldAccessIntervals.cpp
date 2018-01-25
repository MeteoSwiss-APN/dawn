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
#include "test/unit-test/dawn/Optimizer/Passes/TestEnvironment.h"
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

  std::vector<std::shared_ptr<Stencil>> loadTest(std::string sirFilename) {

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

    DAWN_ASSERT_MSG((optimizer->getStencilInstantiationMap().count("compute_extent_test_stencil")),
                    "compute_extent_test_stencil not found in sir");

    return optimizer->getStencilInstantiationMap()["compute_extent_test_stencil"]->getStencils();
  }
};

TEST_F(TestFieldAccessIntervals, test_field_access_interval_01) {
  auto stencils = loadTest("test_field_access_interval_01.sir");
  ASSERT_TRUE((stencils.size() == 1));
  std::shared_ptr<Stencil> stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 2));
  ASSERT_TRUE((stencil->getStage(0)->getExtents() == Extents{-1, 1, -1, 1, 0, 0}));
  ASSERT_TRUE((stencil->getStage(1)->getExtents() == Extents{0, 0, 0, 0, 0, 0}));

  for(auto fieldPair : stencil->getMultiStages().front()->getFields()) {
    Field& field = fieldPair.second;
    int AccessID = fieldPair.first;

    if(AccessID == stencil->getStencilInstantiation().getAccessIDFromName("lap")) {
      ASSERT_TRUE(field.interval_ == Interval(sir::Interval::Start, sir::Interval::End, 11, 0));
    }
    if(AccessID == stencil->getStencilInstantiation().getAccessIDFromName("out") ||
       AccessID == stencil->getStencilInstantiation().getAccessIDFromName("u")) {
      ASSERT_TRUE(field.interval_ == Interval(sir::Interval::Start, sir::Interval::End, 0, 0));
    }
  }
}

TEST_F(TestFieldAccessIntervals, test_field_access_interval_02) {
  auto stencils = loadTest("test_field_access_interval_02.sir");
  ASSERT_TRUE((stencils.size() == 1));
  std::shared_ptr<Stencil> stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 2));
  ASSERT_TRUE((stencil->getStage(0)->getExtents() == Extents{-1, 1, -1, 1, 0, 0}));
  ASSERT_TRUE((stencil->getStage(1)->getExtents() == Extents{0, 0, 0, 0, 0, 0}));

  for(auto fieldPair : stencil->getMultiStages().front()->getFields()) {
    Field& field = fieldPair.second;
    int AccessID = fieldPair.first;
    if(AccessID == stencil->getStencilInstantiation().getAccessIDFromName("lap")) {
      ASSERT_TRUE(field.interval_ == Interval(sir::Interval::Start + 11, sir::Interval::End));
    }
    if(AccessID == stencil->getStencilInstantiation().getAccessIDFromName("out") ||
       AccessID == stencil->getStencilInstantiation().getAccessIDFromName("u")) {
      ASSERT_TRUE(field.interval_ == Interval(sir::Interval::Start, sir::Interval::End));
    }
    if(AccessID == stencil->getStencilInstantiation().getAccessIDFromName("coeff")) {
      ASSERT_TRUE(field.interval_ == Interval(sir::Interval::Start, sir::Interval::End, 11, 0));
      ASSERT_TRUE(field.getAccessedInterval() ==
                  Interval(sir::Interval::Start, sir::Interval::End, 11, 1));
    }
  }
}

TEST_F(TestFieldAccessIntervals, test_field_access_interval_03) {
  auto stencils = loadTest("test_field_access_interval_03.sir");
  ASSERT_TRUE((stencils.size() == 1));
  std::shared_ptr<Stencil> stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 3));
  ASSERT_TRUE((stencil->getStage(0)->getExtents() == Extents{-1, 1, -1, 1, 0, 0}));
  ASSERT_TRUE((stencil->getStage(1)->getExtents() == Extents{0, 0, 0, 0, 0, 0}));
  ASSERT_TRUE((stencil->getStage(2)->getExtents() == Extents{0, 0, 0, 0, 0, 0}));

  for(auto fieldPair : stencil->getMultiStages().front()->getFields()) {
    Field& field = fieldPair.second;
    int AccessID = fieldPair.first;
    if(AccessID == stencil->getStencilInstantiation().getAccessIDFromName("lap")) {
      ASSERT_TRUE(field.interval_ == Interval(sir::Interval::Start + 11, sir::Interval::End));
    }
    if(AccessID == stencil->getStencilInstantiation().getAccessIDFromName("out") ||
       AccessID == stencil->getStencilInstantiation().getAccessIDFromName("u")) {
      ASSERT_TRUE(field.interval_ == Interval(sir::Interval::Start, sir::Interval::End));
    }
    if(AccessID == stencil->getStencilInstantiation().getAccessIDFromName("coeff")) {
      ASSERT_TRUE(field.interval_ == Interval(sir::Interval::Start, sir::Interval::End, 4, 0));
      ASSERT_TRUE(field.getAccessedInterval() ==
                  Interval(sir::Interval::Start, sir::Interval::End, 2, 1));
    }
  }
}

} // anonymous namespace
