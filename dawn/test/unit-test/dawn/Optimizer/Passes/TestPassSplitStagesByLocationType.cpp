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

#include "dawn/AST/LocationType.h"
#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/Compiler/Options.h"
#include "dawn/IIR/ASTFwd.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/PassLocalVarType.h"
#include "dawn/Optimizer/PassRemoveScalars.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Unittest/CompilerUtil.h"
#include "dawn/Unittest/UnittestStmtSimplifier.h"
#include "dawn/Unittest/UnittestUtils.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"

#include <fstream>
#include <gtest/gtest.h>
#include <memory>

using namespace dawn;
using namespace sirgen;

namespace {

class TestPassSplitStageByLocationType : public ::testing::Test {
protected:
  dawn::OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_;
  std::shared_ptr<iir::StencilInstantiation> instantiation_;

  virtual void SetUp() { options_.StageMerger = options_.MergeDoMethods = true; }

  void runPass(const std::string& filename) {
    dawn::UIDGenerator::getInstance()->reset();
    instantiation_ = CompilerUtil::load(filename, options_, context_, TestEnvironment::path_);

    CompilerUtil::runPass<dawn::PassLocalVarType>(context_, instantiation_);
    CompilerUtil::runPass<dawn::PassRemoveScalars>(context_, instantiation_);

    auto const& multistage = instantiation_->getStencils()[0]->getChild(0);
    for(auto const& stage : multistage->getChildren()) {
      DAWN_ASSERT(stage->getLocationType().has_value());
    }

    // ASSERT_TRUE(CompilerUtil::runPass<dawn::PassSplitStageByLocationType>(context_,
    // instantiation));
  }
};

TEST_F(TestPassSplitStageByLocationType, CopyTwoLocationType) {
  // field(cells) in_cell, out_cell;
  // field(edges) in_edge, out_edge;
  // out_cell = in_cell;
  // out_edge = in_edge;

  runPass("input/SplitStageByLocationType/split_stage_by_location_type_test_stencil_01.sir");

  auto const& multistage = instantiation_->getStencils()[0]->getChild(0);
  int expectedNumStages = 2;
  EXPECT_EQ(expectedNumStages, multistage->getChildren().size());

  for(auto const& stage : multistage->getChildren()) {
    ASSERT_TRUE(stage->getLocationType().has_value());
    EXPECT_EQ(stage->getChildren().size(), 1); // only 1 DoMethod
    EXPECT_EQ(stage->getChild(0)->getAST().getStatements().size(),
              1); // only 1 stage in this DoMethod
    auto firstStatement = getNthStmt(*stage->getChild(0), 0);
    if(stage->getLocationType() == ast::LocationType::Cells) {
      // check assignmnt is out_cell = in_cell
      ASSERT_TRUE(firstStatement->equals(expr(assign(field("out_cell"), field("in_cell"))).get(),
                                         /*compareData = */ false));
    }
    if(stage->getLocationType() == ast::LocationType::Edges) {
      // check assignmnt is out_edge = in_edge
      ASSERT_TRUE(firstStatement->equals(expr(assign(field("out_edge"), field("in_edge"))).get(),
                                         /*compareData = */ false));
    }
  }
}

} // anonymous namespace
