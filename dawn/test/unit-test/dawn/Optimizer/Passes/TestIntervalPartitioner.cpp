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
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Optimizer/PassIntervalPartitioner.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

class TestIntervalPartitioner : public ::testing::Test {
  std::unique_ptr<OptimizerContext> context_;

protected:
  virtual void SetUp() {
    dawn::DiagnosticsEngine diag;
    std::shared_ptr<SIR> sir = std::make_shared<SIR>(ast::GridType::Cartesian);
    dawn::OptimizerContext::OptimizerContextOptions options;
    options.PartitionIntervals = true;
    context_ = std::make_unique<OptimizerContext>(diag, options, sir);
  }

  const std::unique_ptr<OptimizerContext>& getContext() { return context_; }

  const std::shared_ptr<iir::StencilInstantiation> loadTest(std::string iirFilename) {
    std::string filename = TestEnvironment::path_;
    if(!filename.empty())
      filename += "/";

    filename += iirFilename;
    std::ifstream file(filename);
    DAWN_ASSERT_MSG((file.good()), std::string("File " + filename + " does not exist").c_str());

    std::string jsonstr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::shared_ptr<iir::StencilInstantiation> stencilInstantion =
        IIRSerializer::deserializeFromString(jsonstr, context_.get());

    return stencilInstantion;
  }
};

TEST_F(TestIntervalPartitioner, test_interval_partition) {
  const std::shared_ptr<iir::StencilInstantiation>& instantiation =
      loadTest("test_interval_partition.iir");
  const std::unique_ptr<OptimizerContext>& context = getContext();

  std::unordered_set<iir::Interval> expected;
  expected.insert(iir::Interval{sir::Interval::Start, sir::Interval::Start});
  expected.insert(iir::Interval{sir::Interval::Start + 1, sir::Interval::Start + 2});
  expected.insert(iir::Interval{sir::Interval::Start + 3, sir::Interval::End - 4});
  expected.insert(iir::Interval{sir::Interval::End - 3, sir::Interval::End - 2});
  expected.insert(iir::Interval{sir::Interval::End - 1, sir::Interval::End});

  PassIntervalPartitioner pass(*context);
  bool result = pass.run(instantiation);
  ASSERT_TRUE(result);

  const auto& stencils = instantiation->getIIR()->getChildren();
  ASSERT_TRUE((stencils.size() == 1));
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];
  ASSERT_TRUE(stencil->getChildren().size() > 0);
  ASSERT_TRUE(stencil->getNumStages() == 3);

  const auto& multiStage = stencil->getChildren().begin()->get();
  std::unordered_set<iir::Interval> intervals = multiStage->getIntervals();

  ASSERT_TRUE(intervals.size() == expected.size());
  for(const auto& interval : expected) {
    ASSERT_TRUE(intervals.find(interval) != intervals.end());
  }
}

} // anonymous namespace
