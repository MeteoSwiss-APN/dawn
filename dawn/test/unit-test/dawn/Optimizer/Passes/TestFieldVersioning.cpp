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
#include "dawn/Optimizer/PassFieldVersioning.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Unittest/CompilerUtil.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

class TestFieldVersioning : public ::testing::Test {
protected:
  dawn::OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_;

  void raceConditionTest(const std::string& filename) {
    std::shared_ptr<iir::StencilInstantiation> instantiation;
    CompilerUtil::load(filename, options_, context_, instantiation, TestEnvironment::path_);

    PassFieldVersioning pass(*context_);
    bool result = pass.run(instantiation);
    ASSERT_FALSE(result); // Expect pass to fail...

    DiagnosticsEngine& diag = context_->getDiagnostics();
    ASSERT_TRUE(diag.hasErrors());
  }

  void versioningTest(const std::string& filename) {
    std::shared_ptr<iir::StencilInstantiation> instantiation;
    CompilerUtil::load(filename, options_, context_, instantiation, TestEnvironment::path_);

    PassFieldVersioning pass(*context_);
    bool result = pass.run(instantiation);
    ASSERT_TRUE(result); // Expect pass to succeed...
  }
};

TEST_F(TestFieldVersioning, RaceCondition1) { raceConditionTest("RaceCondition01.iir"); }

TEST_F(TestFieldVersioning, RaceCondition2) { raceConditionTest("RaceCondition02.iir"); }

TEST_F(TestFieldVersioning, RaceCondition3) { raceConditionTest("RaceCondition03.sir"); }

TEST_F(TestFieldVersioning, VersioningTest1) { versioningTest("VersioningTest01.iir"); }

} // anonymous namespace
