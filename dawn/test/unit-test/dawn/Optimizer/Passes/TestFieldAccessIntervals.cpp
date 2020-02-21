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
#include "dawn/Serialization/IIRSerializer.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"
#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

TEST(TestFieldAccessIntervals, test_field_access_interval_01) {
  /*
    interval k_flat = k_start + 12;
    vertical_region(k_start, k_flat) { in = 12; }
  */
  auto stencilInstantiation = IIRSerializer::deserialize("input/test_field_access_interval_01.iir");
  const auto& metadata = stencilInstantiation->getMetaData();
  const auto& stencils = stencilInstantiation->getStencils();
  EXPECT_EQ(stencils.size(), 1);

  int inID = metadata.getAccessIDFromName("in");
  auto fieldToIDMap = stencils[0]->getFields();
  auto field = fieldToIDMap.at(inID).field;
  EXPECT_EQ(field.getInterval(),
            (iir::Interval{sir::Interval::Start, sir::Interval::Start, 0, 12}));
}

TEST(TestFieldAccessIntervals, test_field_access_interval_02) {
  /*
    vertical_region(k_start, k_start) { in = 12; }
    vertical_region(k_end, k_end) { in = 10; }
  */
  auto stencilInstantiation = IIRSerializer::deserialize("input/test_field_access_interval_02.iir");
  const auto& metadata = stencilInstantiation->getMetaData();
  const auto& stencils = stencilInstantiation->getStencils();
  EXPECT_EQ(stencils.size(), 1);

  int inID = metadata.getAccessIDFromName("in");
  auto fieldToIDMap = stencils[0]->getFields();
  auto field = fieldToIDMap.at(inID).field;
  EXPECT_EQ(field.getInterval(), (iir::Interval{sir::Interval::Start, sir::Interval::End, 0, 0}));
}
TEST(TestFieldAccessIntervals, test_field_access_interval_03) {
  /*
    vertical_region(k_start, k_start + 2) { in = 12; }
  */
  auto stencilInstantiation = IIRSerializer::deserialize("input/test_field_access_interval_03.iir");
  const auto& metadata = stencilInstantiation->getMetaData();
  const auto& stencils = stencilInstantiation->getStencils();
  EXPECT_EQ(stencils.size(), 1);

  int inID = metadata.getAccessIDFromName("in");
  auto fieldToIDMap = stencils[0]->getFields();
  auto field = fieldToIDMap.at(inID).field;
  EXPECT_EQ(field.getInterval(), (iir::Interval{sir::Interval::Start, sir::Interval::Start, 0, 2}));
}

} // anonymous namespace
