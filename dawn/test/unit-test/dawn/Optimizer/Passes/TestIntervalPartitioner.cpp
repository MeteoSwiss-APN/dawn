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
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"
#include <fstream>
#include <gtest/gtest.h>
#include <streambuf>

using namespace dawn;

namespace {

class TestIntervalPartitioner : public ::testing::Test {
  std::unique_ptr<dawn::Options> compileOptions_;
  dawn::DawnCompiler compiler_;

protected:
  TestIntervalPartitioner() : compiler_(compileOptions_.get()) {}
  virtual void SetUp() {}

  const std::shared_ptr<iir::StencilInstantiation> loadTest(std::string sirFilename) {
    // std::string filename = TestEnvironment::path_ + "/" + sirFilename;
    std::string filename = sirFilename;
    std::ifstream file(filename);
    DAWN_ASSERT_MSG((file.good()), std::string("File " + filename + " does not exists").c_str());

    std::string jsonstr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::shared_ptr<SIR> sir =
        SIRSerializer::deserializeFromString(jsonstr, SIRSerializer::Format::Json);

    std::unique_ptr<OptimizerContext> optimizer = compiler_.runOptimizer(sir);
    // Report diganostics
    if(compiler_.getDiagnostics().hasDiags()) {
      for(const auto& diag : compiler_.getDiagnostics().getQueue())
        std::cerr << "Compilation Error " << diag->getMessage() << std::endl;
      throw std::runtime_error("compilation failed");
    }

    DAWN_ASSERT_MSG((optimizer->getStencilInstantiationMap().count("interval_partition_test")),
                    "interval_partition_test not found in sir");

    return optimizer->getStencilInstantiationMap()["interval_partition_test"];
  }
};

TEST_F(TestIntervalPartitioner, test_interval_partition) {
  const std::shared_ptr<iir::StencilInstantiation>& instantiation =
      loadTest("test_interval_partition.sir");
  // const auto& metadata = instantiation->getMetaData();
  const std::unique_ptr<iir::IIR>& IIR = instantiation->getIIR();

  const auto& stencils = IIR->getChildren();
  ASSERT_TRUE((stencils.size() == 1));
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];
  ASSERT_TRUE(stencil->getNumStages() == 3);

  auto multiStage = stencil->getChildren().begin()->get();
  std::unordered_set<iir::Interval> intervals = multiStage->getIntervals();
  // Actually think this should be 5, more work to do...
  ASSERT_TRUE(intervals.size() == 4);
}

} // anonymous namespace
