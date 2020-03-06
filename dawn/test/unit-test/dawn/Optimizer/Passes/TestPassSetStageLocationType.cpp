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
#include "dawn/Optimizer/PassSetStageLocationType.h"
#include "dawn/Optimizer/PassSplitStageFineGrained.h"
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

class TestPassSetStageLocationType : public ::testing::Test {
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
    CompilerUtil::runPass<dawn::PassSplitStageFineGrained>(context_, instantiation_);

    ASSERT_TRUE(CompilerUtil::runPass<dawn::PassSetStageLocationType>(context_, instantiation_));
  }
};

TEST_F(TestPassSetStageLocationType, CopyFieldsLocationTypes) {
  // field(cells) in_cell, out_cell;
  // field(edges) in_edge, out_edge;
  // fields(vertices) in_vertex, out_vertex
  // out_cell = in_cell;
  // out_edge = in_edge;
  // out_vertex = in_vertex;

  runPass("samples/test_set_stage_location_type_copy_fields.sir");

  auto const& multistage = instantiation_->getStencils()[0]->getChild(0);

  auto const& firstStage = multistage->getChild(0);
  ASSERT_EQ(firstStage->getLocationType(), ast::LocationType::Cells);

  auto const& secondStage = multistage->getChild(1);
  ASSERT_EQ(secondStage->getLocationType(), ast::LocationType::Edges);

  auto const& thirdStage = multistage->getChild(3);
  ASSERT_EQ(thirdStage->getLocationType(), ast::LocationType::Vertices);
}

TEST_F(TestPassSetStageLocationType, CopyVarsLocationTypes) {
  // field(cells) in_cell;
  // field(edges) in_edge;
  // field(vertices) in_vertex;
  // var out_var_cell;
  // var out_var_edge;
  // var out_var_vertex;
  // out_var_cell = in_cell;
  // out_var_edge = in_edge;
  // out_var_vertex = in_vertex;

  runPass("samples/test_set_stage_location_type_copy_vars.sir");

  auto const& multistage = instantiation_->getStencils()[0]->getChild(0);

  const auto& var_decl_cell = multistage->getChild(0);
  ASSERT_EQ(ast::LocationType::Cells, var_decl_cell->getLocationType());

  const auto& var_decl_edge = multistage->getChild(1);
  ASSERT_EQ(ast::LocationType::Edges, var_decl_edge->getLocationType());

  const auto& var_decl_vertex = multistage->getChild(2);
  ASSERT_EQ(ast::LocationType::Vertices, var_decl_vertex->getLocationType());

  const auto& assign_cell = multistage->getChild(3);
  ASSERT_EQ(ast::LocationType::Cells, assign_cell->getLocationType());

  const auto& assign_edge = multistage->getChild(4);
  ASSERT_EQ(ast::LocationType::Edges, assign_edge->getLocationType());

  const auto& assign_vertex = multistage->getChild(5);
  ASSERT_EQ(ast::LocationType::Vertices, assign_vertex->getLocationType());
}

TEST_F(TestPassSetStageLocationType, IfStmt) {
  // field(cells) in_cell;
  // var out_var_cell;
  // if(out_var_cell) out_var_cell = in_cell;

  runPass("samples/test_set_stage_location_type_if_stmt.sir");

  auto const& multistage = instantiation_->getStencils()[0]->getChild(0);

  const auto& var_decl_cell = multistage->getChild(0);
  ASSERT_EQ(ast::LocationType::Cells, var_decl_cell->getLocationType());

  const auto& if_cell = multistage->getChild(1);
  ASSERT_EQ(ast::LocationType::Cells, if_cell->getLocationType());
}

TEST_F(TestPassSetStageLocationType, FunctionCall) {
  // stencil_function f(field(cells) out) {
  //  out = 2.0;
  // }
  //
  // fields(cells) out_cell;
  // f(out_cells);

  runPass("samples/test_set_stage_location_type_function_call.sir");
  // TODO
}

} // namespace
