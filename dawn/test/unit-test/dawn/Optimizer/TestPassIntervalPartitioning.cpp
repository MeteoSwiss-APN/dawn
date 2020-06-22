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

#include "dawn/IIR/IIR.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/PassIntervalPartitioning.h"
#include "dawn/Serialization/IIRSerializer.h"

#include <fstream>
#include <gtest/gtest.h>

namespace {

using namespace dawn;

TEST(TestPassIntervalPartitioning, test_interval_partition) {
  std::unordered_set<iir::Interval> expected;

  dawn::UIDGenerator::getInstance()->reset();
  expected.insert(iir::Interval{sir::Interval::Start, sir::Interval::Start});
  expected.insert(iir::Interval{sir::Interval::Start + 1, sir::Interval::Start + 2});
  expected.insert(iir::Interval{sir::Interval::Start + 3, sir::Interval::End - 4});
  expected.insert(iir::Interval{sir::Interval::End - 3, sir::Interval::End - 2});
  expected.insert(iir::Interval{sir::Interval::End - 1, sir::Interval::End});

  auto instantiation = IIRSerializer::deserialize("input/test_interval_partition.iir");

  // Expect pass to succeed...
  PassIntervalPartitioning intervalPartitioningPass;
  EXPECT_TRUE(intervalPartitioningPass.run(instantiation));

  const auto& stencils = instantiation->getIIR()->getChildren();
  ASSERT_TRUE((stencils.size() == 1));
  const auto& stencil = stencils[0];
  ASSERT_TRUE(stencil->getNumStages() == 3);

  const auto& multiStage = stencil->getChildren().begin()->get();
  auto intervals = multiStage->getIntervals();

  ASSERT_TRUE(intervals.size() == expected.size());
  for(const auto& interval : expected) {
    ASSERT_TRUE(intervals.find(interval) != intervals.end());
  }
}

} // anonymous namespace
