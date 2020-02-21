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
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"

#include "dawn/Support/DiagnosticsEngine.h"

#include <fstream>
#include <gtest/gtest.h>
#include <memory>

using namespace dawn;

namespace {

class TestPassFieldVersioning : public ::testing::Test {
public:
  TestPassFieldVersioning() {
    context_ = std::make_unique<OptimizerContext>(diagnostics_, options_, nullptr);
  }

protected:
  dawn::OptimizerContext::OptimizerContextOptions options_;
  DiagnosticsEngine diagnostics_;
  std::unique_ptr<OptimizerContext> context_;

  void raceConditionTest(const std::string& filename) {
    context_->getDiagnostics().clear();
    std::shared_ptr<iir::StencilInstantiation> instantiation = IIRSerializer::deserialize(filename);

    // Expect pass to fail...
    dawn::PassFieldVersioning pass(*context_);
    ASSERT_FALSE(pass.run(instantiation));
    ASSERT_TRUE(context_->getDiagnostics().hasErrors());
  }

  void versioningTest(const std::string& filename) {
    context_->getDiagnostics().clear();
    std::shared_ptr<iir::StencilInstantiation> instantiation = IIRSerializer::deserialize(filename);

    // Expect pass to succeed...
    dawn::PassFieldVersioning pass(*context_);
    ASSERT_TRUE(pass.run(instantiation));
  }
};

TEST_F(TestPassFieldVersioning, RaceCondition1) {
  raceConditionTest("input/TestPassFieldVersioning_01.iir");
}

TEST_F(TestPassFieldVersioning, RaceCondition2) {
  raceConditionTest("input/TestPassFieldVersioning_02.iir");
}

TEST_F(TestPassFieldVersioning, RaceCondition3) {
  raceConditionTest("input/TestPassFieldVersioning_03.iir");
}

TEST_F(TestPassFieldVersioning, VersioningTest1) {
  versioningTest("input/TestPassFieldVersioning_04.iir");
}

TEST_F(TestPassFieldVersioning, VersioningTest2) {
  versioningTest("input/TestPassFieldVersioning_05.iir");
}

TEST_F(TestPassFieldVersioning, VersioningTest3) {
  versioningTest("input/TestPassFieldVersioning_06.iir");
}

TEST_F(TestPassFieldVersioning, VersioningTest4) {
  versioningTest("input/TestPassFieldVersioning_07.iir");
}

TEST_F(TestPassFieldVersioning, VersioningTest5) {
  versioningTest("input/TestPassFieldVersioning_08.iir");
}

TEST_F(TestPassFieldVersioning, VersioningTest6) {
  versioningTest("input/TestPassFieldVersioning_09.iir");
}

TEST_F(TestPassFieldVersioning, VersioningTest7) {
  versioningTest("input/TestPassFieldVersioning_10.iir");
}
} // anonymous namespace
