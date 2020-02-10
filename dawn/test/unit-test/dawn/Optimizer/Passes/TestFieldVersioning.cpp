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
#include "dawn/Optimizer/PassInlining.h"
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
    CompilerUtil::clearDiags();
    std::shared_ptr<iir::StencilInstantiation> instantiation =
        CompilerUtil::load(filename, options_, context_, TestEnvironment::path_);

    // Inline pass is a prerequisite...
    ASSERT_TRUE(CompilerUtil::runPass<dawn::PassInlining>(
        context_, instantiation, true, dawn::PassInlining::InlineStrategy::InlineProcedures));

    // Expect pass to fail...
    ASSERT_FALSE(CompilerUtil::runPass<dawn::PassFieldVersioning>(context_, instantiation));
    ASSERT_TRUE(context_->getDiagnostics().hasErrors());
  }

  void versioningTest(const std::string& filename) {
    CompilerUtil::clearDiags();
    std::shared_ptr<iir::StencilInstantiation> instantiation =
        CompilerUtil::load(filename, options_, context_, TestEnvironment::path_);

    // Expect pass to succeed...
    ASSERT_TRUE(CompilerUtil::runPass<dawn::PassFieldVersioning>(context_, instantiation));
  }
};

TEST_F(TestFieldVersioning, RaceCondition1) { raceConditionTest("input/RaceCondition01.sir"); }

TEST_F(TestFieldVersioning, RaceCondition2) { raceConditionTest("input/RaceCondition02.sir"); }

TEST_F(TestFieldVersioning, RaceCondition3) { raceConditionTest("input/RaceCondition03.sir"); }

TEST_F(TestFieldVersioning, VersioningTest1) { versioningTest("input/VersioningTest01.sir"); }

TEST_F(TestFieldVersioning, VersioningTest2) { versioningTest("input/VersioningTest02.sir"); }

TEST_F(TestFieldVersioning, VersioningTest3) { versioningTest("input/VersioningTest03.sir"); }

TEST_F(TestFieldVersioning, VersioningTest4) { versioningTest("input/VersioningTest04.sir"); }

TEST_F(TestFieldVersioning, VersioningTest5) { versioningTest("input/VersioningTest05.sir"); }

TEST_F(TestFieldVersioning, VersioningTest6) { versioningTest("input/VersioningTest06.sir"); }

TEST_F(TestFieldVersioning, VersioningTest7) { versioningTest("input/VersioningTest07.sir"); }

} // anonymous namespace
