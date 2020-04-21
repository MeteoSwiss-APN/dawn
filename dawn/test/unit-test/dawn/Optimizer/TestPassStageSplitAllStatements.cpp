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

#include "dawn/IIR/ASTFwd.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassStageSplitAllStatements.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Support/Type.h"
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
  OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_;
  std::shared_ptr<iir::StencilInstantiation> instantiation_;

  void runPass(const std::string& filename) {
    UIDGenerator::getInstance()->reset();
    instantiation_ = CompilerUtil::load(filename, options_, context_, TestEnvironment::path_);

    ASSERT_TRUE(CompilerUtil::runPass<dawn::PassStageSplitAllStatements>(context_, instantiation_));
  }
};

TEST_F(TestPassStageSplitAllStatements, NoStmt) {
  runPass("input/test_stage_split_all_statements_no_stmt.iir");

  auto const& multistage = instantiation_->getStencils()[0]->getChild(0);
  ASSERT_EQ(1, multistage->getChildren().size());
  ASSERT_EQ(0, multistage->getChild(0)->getSingleDoMethod().getAST().getStatements().size());
}

TEST_F(TestPassStageSplitAllStatements, OneStmt) {
  // var a;
  runPass("input/test_stage_split_all_statements_one_stmt.iir");

  auto const& multistage = instantiation_->getStencils()[0]->getChild(0);
  ASSERT_EQ(1, multistage->getChildren().size());
  auto firstDoMethod = multistage->getChild(0)->getSingleDoMethod().getAST();
  ASSERT_EQ(1, firstDoMethod.getStatements().size());
  ASSERT_TRUE(firstDoMethod.getStatements()[0]->equals(vardecl("a", BuiltinTypeID::Integer).get(),
                                                       /*compareData = */ false));
}

TEST_F(TestPassStageSplitAllStatements, TwoStmts) {
  // var a;
  // var b;
  runPass("input/test_stage_split_all_statements_two_stmt.iir");

  auto const& multistage = instantiation_->getStencils()[0]->getChild(0);
  ASSERT_EQ(2, multistage->getChildren().size());
  auto firstDoMethod = multistage->getChild(0)->getSingleDoMethod().getAST();
  ASSERT_EQ(1, firstDoMethod.getStatements().size());
  ASSERT_TRUE(firstDoMethod.getStatements()[0]->equals(vardecl("a", BuiltinTypeID::Integer).get(),
                                                       /*compareData = */ false));
  auto secondDoMethod = multistage->getChild(1)->getSingleDoMethod().getAST();
  ASSERT_EQ(1, secondDoMethod.getStatements().size());
  ASSERT_TRUE(secondDoMethod.getStatements()[0]->equals(vardecl("b", BuiltinTypeID::Integer).get(),
                                                        /*compareData = */ false));
}

} // namespace
