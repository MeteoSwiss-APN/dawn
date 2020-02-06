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
#include "dawn/Optimizer/PassInlining.h"
#include "dawn/Optimizer/PassValidation.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Support/Exception.h"
#include "dawn/Support/FileUtil.h"
#include "dawn/Unittest/CompilerUtil.h"
#include "dawn/Validator/IntegrityChecker.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

TEST(TestIntegrityChecker, GlobalsOptimizedAway) {
  // Load IIR from file
  std::unique_ptr<OptimizerContext> context;
  dawn::OptimizerContext::OptimizerContextOptions options;
  std::shared_ptr<iir::StencilInstantiation> instantiation =
      CompilerUtil::load("input/globals_opt_away.sir", options, context);

  // Run parallel group
  ASSERT_TRUE(CompilerUtil::runGroup(PassGroup::Parallel, context, instantiation));

  // Run validation pass and check for exception
  IntegrityChecker checker(instantiation.get());
  try {
    checker.run();
    FAIL() << "Semantic error not thrown";
  } catch(SemanticError& error) {
    SUCCEED();
  }
}

} // anonymous namespace
