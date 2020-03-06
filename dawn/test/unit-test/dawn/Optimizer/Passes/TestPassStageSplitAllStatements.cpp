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
#include "dawn/IIR/ASTFwd.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/PassStageSplitAllStatements.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Unittest/ASTConstructionAliases.h"
#include "dawn/Unittest/CompilerUtil.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"

#include <fstream>
#include <gtest/gtest.h>
#include <memory>

using namespace dawn;
using namespace astgen;

namespace {

class TestPassStageSplitAllStatements : public ::testing::Test {
protected:
  dawn::OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_;
  std::shared_ptr<iir::StencilInstantiation> instantiation_;

  virtual void SetUp() { options_.StageMerger = options_.MergeDoMethods = true; }

  void runPass(const std::string& filename) {
    dawn::UIDGenerator::getInstance()->reset();
    instantiation_ = CompilerUtil::load(filename, options_, context_, TestEnvironment::path_);

    CompilerUtil::write(context_, 0, 100, "test");

    ASSERT_TRUE(CompilerUtil::runPass<dawn::PassStageSplitAllStatements>(context_, instantiation_));
  }
};

TEST_F(TestPassStageSplitAllStatements, NoStmt) {
  runPass("input/test_stage_split_all_statements_no_stmt.sir");
}

TEST_F(TestPassStageSplitAllStatements, OneStmt) {
  runPass("input/test_stage_split_all_statements_one_stmt.sir");
}

} // namespace
