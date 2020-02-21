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
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"
#include <fstream>
#include <gtest/gtest.h>
#include <optional>
#include <streambuf>

using namespace dawn;

namespace {

TEST(TestFieldAccessIntervals, test_field_access_interval_01) {
  auto stencilInstantiation = IIRSerializer::deserialize("input/test_field_access_interval_01.iir");
  const auto& metadata = stencilInstantiation->getMetaData();
  const auto& stencils = stencilInstantiation->getStencils();
  EXPECT_EQ(stencils.size(), 1);
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getNumStages(), 2);
  EXPECT_EQ(stencil->getStage(0)->getExtents(), (iir::Extents(ast::cartesian, -1, 1, -1, 1, 0, 0)));
  EXPECT_EQ(stencil->getStage(1)->getExtents(), (iir::Extents(ast::cartesian, 0, 0, 0, 0, 0, 0)));

  for(auto fieldPair : (*stencil->childrenBegin())->getFields()) {
    iir::Field& field = fieldPair.second;
    int AccessID = fieldPair.first;

    if(AccessID == metadata.getAccessIDFromName("lap")) {
      EXPECT_EQ(field.getInterval(),
                (iir::Interval{sir::Interval::Start, sir::Interval::End, 11, 0}));
    }
    if(AccessID == metadata.getAccessIDFromName("out") ||
       AccessID == metadata.getAccessIDFromName("u")) {
      EXPECT_EQ(field.getInterval(),
                (iir::Interval{sir::Interval::Start, sir::Interval::End, 0, 0}));
    }
  }
}

TEST(TestFieldAccessIntervals, test_field_access_interval_02) {
  auto stencilInstantiation = IIRSerializer::deserialize("input/test_field_access_interval_02.iir");
  const auto& metadata = stencilInstantiation->getMetaData();
  const auto& stencils = stencilInstantiation->getStencils();
  EXPECT_EQ(stencils.size(), 1);
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getNumStages(), 2);
  EXPECT_EQ(stencil->getStage(0)->getExtents(),
            ((iir::Extents(ast::cartesian, -1, 1, -1, 1, 0, 0))));
  EXPECT_EQ(stencil->getStage(1)->getExtents(), ((iir::Extents(ast::cartesian, 0, 0, 0, 0, 0, 0))));

  for(auto fieldPair : (*stencil->childrenBegin())->getFields()) {
    iir::Field& field = fieldPair.second;
    int AccessID = fieldPair.first;
    if(AccessID == metadata.getAccessIDFromName("lap")) {
      EXPECT_EQ(field.getInterval(),
                (iir::Interval{sir::Interval::Start + 11, sir::Interval::End}));
    }
    if(AccessID == metadata.getAccessIDFromName("out") ||
       AccessID == metadata.getAccessIDFromName("u")) {
      EXPECT_EQ(field.getInterval(), (iir::Interval{sir::Interval::Start, sir::Interval::End}));
    }
    if(AccessID == metadata.getAccessIDFromName("coeff")) {
      EXPECT_EQ(field.getInterval(),
                (iir::Interval{sir::Interval::Start, sir::Interval::End, 11, 0}));
      EXPECT_EQ(field.computeAccessedInterval(), (iir::Interval{12, sir::Interval::End + 1}));
    }
  }
}

TEST(TestFieldAccessIntervals, test_field_access_interval_03) {
  auto stencilInstantiation = IIRSerializer::deserialize("input/test_field_access_interval_03.iir");
  const auto& metadata = stencilInstantiation->getMetaData();
  const auto& stencils = stencilInstantiation->getStencils();
  EXPECT_EQ(stencils.size(), 1);
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getNumStages(), 3);
  EXPECT_EQ(stencil->getStage(0)->getExtents(), (iir::Extents(ast::cartesian, -1, 1, -1, 1, 0, 0)));
  EXPECT_EQ(stencil->getStage(1)->getExtents(), (iir::Extents(ast::cartesian, 0, 0, 0, 0, 0, 0)));
  EXPECT_EQ(stencil->getStage(2)->getExtents(), (iir::Extents(ast::cartesian, 0, 0, 0, 0, 0, 0)));

  for(auto fieldPair : (*stencil->childrenBegin())->getFields()) {
    iir::Field& field = fieldPair.second;
    int AccessID = fieldPair.first;
    if(AccessID == metadata.getAccessIDFromName("lap")) {
      EXPECT_EQ(field.getInterval(),
                (iir::Interval{sir::Interval::Start + 11, sir::Interval::End}));
    }
    if(AccessID == metadata.getAccessIDFromName("out") ||
       AccessID == metadata.getAccessIDFromName("u")) {
      EXPECT_EQ(field.getInterval(), (iir::Interval{sir::Interval::Start, sir::Interval::End}));
    }
    if(AccessID == metadata.getAccessIDFromName("coeff")) {
      EXPECT_EQ(field.getInterval(),
                (iir::Interval{sir::Interval::Start, sir::Interval::End, 4, 0}));
      EXPECT_EQ(field.computeAccessedInterval(),
                (iir::Interval{sir::Interval::Start, sir::Interval::End, 2, 1}));
    }
  }
}

TEST(TestFieldAccessIntervals, test_field_access_interval_04) {
  auto stencilInstantiation = IIRSerializer::deserialize("input/test_field_access_interval_04.iir");
  const auto& stencils = stencilInstantiation->getStencils();
  EXPECT_EQ(stencils.size(), 1);
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getNumStages(), 3);

  iir::MultiStage& multiStage = **(stencil->childrenBegin());

  std::optional<iir::Interval> enclosingInterval =
      multiStage.getEnclosingAccessIntervalTemporaries();
  ASSERT_TRUE(enclosingInterval.has_value());
  EXPECT_EQ((*enclosingInterval), (iir::Interval{2, 14}));
}

TEST(TestFieldAccessIntervals, test_field_access_interval_05) {
  auto stencilInstantiation = IIRSerializer::deserialize("input/test_field_access_interval_05.iir");
  const auto& metadata = stencilInstantiation->getMetaData();
  const auto& stencils = stencilInstantiation->getStencils();
  EXPECT_EQ(stencils.size(), 1);
  const std::unique_ptr<iir::Stencil>& stencil = stencils[0];

  EXPECT_EQ(stencil->getNumStages(), 2);
  EXPECT_EQ(stencil->getStage(0)->getExtents(), (iir::Extents(ast::cartesian, -1, 1, -1, 1, 0, 0)));
  EXPECT_EQ(stencil->getStage(1)->getExtents(), (iir::Extents(ast::cartesian, 0, 0, 0, 0, 0, 0)));

  for(auto fieldPair : (*stencil->childrenBegin())->getFields()) {
    iir::Field& field = fieldPair.second;
    int AccessID = fieldPair.first;
    if(AccessID == metadata.getAccessIDFromName("lap")) {
      EXPECT_EQ(field.getInterval(),
                (iir::Interval{sir::Interval::Start + 11, sir::Interval::End}));
      EXPECT_EQ(field.computeAccessedInterval(),
                (iir::Interval{sir::Interval::Start + 10, sir::Interval::End}));
    }
    if(AccessID == metadata.getAccessIDFromName("out") ||
       AccessID == metadata.getAccessIDFromName("u")) {
      EXPECT_EQ(field.getInterval(), (iir::Interval{sir::Interval::Start, sir::Interval::End}));
    }
    if(AccessID == metadata.getAccessIDFromName("coeff")) {
      EXPECT_EQ(field.getInterval(),
                (iir::Interval{sir::Interval::Start, sir::Interval::End, 11, 0}));
      EXPECT_EQ(field.computeAccessedInterval(), (iir::Interval{12, sir::Interval::End + 1}));
    }
  }
}

} // anonymous namespace
