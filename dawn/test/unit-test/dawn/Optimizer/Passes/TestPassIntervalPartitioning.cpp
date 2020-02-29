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
#include "dawn/Optimizer/PassIntervalPartitioning.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"

#include <fstream>
#include <gtest/gtest.h>

namespace {

using namespace dawn;

class TestPassIntervalPartitioning : public ::testing::Test {
protected:
  dawn::OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_;
  std::unordered_set<iir::Interval> expected_;

  virtual void SetUp() {
    options_.IntervalPartitioning = true;
    dawn::UIDGenerator::getInstance()->reset();
    expected_.insert(iir::Interval{sir::Interval::Start, sir::Interval::Start});
    expected_.insert(iir::Interval{sir::Interval::Start + 1, sir::Interval::Start + 2});
    expected_.insert(iir::Interval{sir::Interval::Start + 3, sir::Interval::End - 4});
    expected_.insert(iir::Interval{sir::Interval::End - 3, sir::Interval::End - 2});
    expected_.insert(iir::Interval{sir::Interval::End - 1, sir::Interval::End});
  }
};

TEST_F(TestPassIntervalPartitioning, test_interval_partition) {
  std::string filepath = "input/test_interval_partition.iir";
  if(!TestEnvironment::path_.empty())
    filepath = TestEnvironment::path_ + "/" + filepath;
  auto instantiation = IIRSerializer::deserialize(filepath);

  // Expect pass to succeed...
  PassIntervalPartitioning intervalPartitioningPass(*context_);
  EXPECT_TRUE(intervalPartitioningPass.run(instantiation));

  const auto& stencils = instantiation->getIIR()->getChildren();
  ASSERT_TRUE((stencils.size() == 1));
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];
  ASSERT_TRUE(stencil->getNumStages() == 3);

  const auto& multiStage = stencil->getChildren().begin()->get();
  std::unordered_set<iir::Interval> intervals = multiStage->getIntervals();

  ASSERT_TRUE(intervals.size() == expected_.size());
  for(const auto& interval : expected_) {
    ASSERT_TRUE(intervals.find(interval) != intervals.end());
  }
}

} // anonymous namespace
