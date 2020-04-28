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
#include "dawn/Optimizer/MultiStageChecker.h"
#include "dawn/Optimizer/PassSetDependencyGraph.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Support/Exception.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

TEST(TestMultiStageChecker, LaplacianTwoStep) {
  dawn::DiagnosticsEngine diag;
  dawn::OptimizerContext::OptimizerContextOptions options;
  auto sir = std::make_shared<SIR>(ast::GridType::Cartesian);

  // Load IIR from file
  auto instantiation = IIRSerializer::deserialize("input/LaplacianTwoStep.iir");

  // Run dependency graph pass
  auto context = std::make_unique<OptimizerContext>(diag, options, sir);
  PassSetDependencyGraph dependencyGraphPass(*context);
  EXPECT_TRUE(dependencyGraphPass.run(instantiation));

  MultiStageChecker checker(instantiation.get(), 1);

  // Run multistage checker and succeed if exception is thrown
  try {
    checker.run();
    FAIL() << "Max halo error not caught";
  } catch(CompileError& error) {
    SUCCEED();
  }
}

} // anonymous namespace
