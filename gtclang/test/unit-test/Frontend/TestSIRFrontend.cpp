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

#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/SIRSerializer.h"
#include <fstream>
#include <gtest/gtest.h>

namespace {
using namespace dawn;

class TestSIRFrontend : public ::testing::Test {
protected:
  void runTest(const std::string& irFilename, const std::string frontend = "gt4py") {
    std::string refPath = irFilename + "_ref.sir";
    auto refSIR = SIRSerializer::deserialize(refPath);

    std::string testPath = irFilename + "_" + frontend + ".sir";
    auto testSIR = SIRSerializer::deserialize(testPath);

    ASSERT_TRUE(*refSIR == *testSIR);
  }
};

TEST_F(TestSIRFrontend, TridiagonalSolve) { runTest("input/tridiagonal_solve"); }

TEST_F(TestSIRFrontend, HorizontalDifference) { runTest("input/horizontal_difference"); }

TEST_F(TestSIRFrontend, Coriolis) { runTest("input/coriolis_stencil"); }

TEST_F(TestSIRFrontend, DISABLED_BurgersDemo) { runTest("input/burgers_demo"); }

} // namespace
