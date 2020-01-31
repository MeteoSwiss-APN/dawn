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
};

TEST_F(TestFieldVersioning, RaceCondition1) {
  const std::shared_ptr<iir::StencilInstantiation>& instantiation =
      CompilerUtil::load("RaceCondition01.iir", options_, context_, TestEnvironment::path_);

  PassFieldVersioning pass(*context_);
  bool result = pass.run(instantiation);
  ASSERT_FALSE(result);                   // Expect pass to fail...

  DiagnosticsEngine& diag = context_->getDiagnostics();
  ASSERT_TRUE(diag.hasErrors());

  const std::string& msg = (*diag.getQueue().begin())->getMessage();
  ASSERT_TRUE(msg.find("race-condition") != std::string::npos);
}

} // anonymous namespace
