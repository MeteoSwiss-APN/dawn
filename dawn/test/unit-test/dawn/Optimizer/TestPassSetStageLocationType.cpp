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
#include "dawn/Compiler/Options.h"
#include "dawn/IIR/ASTFwd.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassLocalVarType.h"
#include "dawn/Optimizer/PassRemoveScalars.h"
#include "dawn/Optimizer/PassSetStageLocationType.h"
#include "dawn/Optimizer/PassStageSplitAllStatements.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Unittest/CompilerUtil.h"
#include "dawn/Unittest/UnittestUtils.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"

#include <fstream>
#include <gtest/gtest.h>
#include <memory>

using namespace dawn;

namespace {

class TestPassSetStageLocationType : public ::testing::Test {
protected:
  OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_;
  std::shared_ptr<iir::StencilInstantiation> instantiation_;

  void runPass(const std::string& filename) {
    dawn::UIDGenerator::getInstance()->reset();
    instantiation_ = CompilerUtil::load(filename, options_, context_, TestEnvironment::path_);

    CompilerUtil::runPass<dawn::PassLocalVarType>(context_, instantiation_);
    CompilerUtil::runPass<dawn::PassRemoveScalars>(context_, instantiation_);
    CompilerUtil::runPass<dawn::PassStageSplitAllStatements>(context_, instantiation_);

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

  runPass("input/test_set_stage_location_type_copy_fields.sir");

  auto const& multistage = instantiation_->getStencils()[0]->getChild(0);

  auto const& firstStage = multistage->getChild(0);
  ASSERT_EQ(firstStage->getLocationType(), ast::LocationType::Cells);

  auto const& secondStage = multistage->getChild(1);
  ASSERT_EQ(secondStage->getLocationType(), ast::LocationType::Edges);

  auto const& thirdStage = multistage->getChild(2);
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

  runPass("input/test_set_stage_location_type_copy_vars.sir");

  auto const& multistage = instantiation_->getStencils()[0]->getChild(0);

  const auto& varDeclCell = multistage->getChild(0);
  ASSERT_EQ(ast::LocationType::Cells, varDeclCell->getLocationType());

  const auto& varDeclEdge = multistage->getChild(1);
  ASSERT_EQ(ast::LocationType::Edges, varDeclEdge->getLocationType());

  const auto& varDeclVertex = multistage->getChild(2);
  ASSERT_EQ(ast::LocationType::Vertices, varDeclVertex->getLocationType());

  const auto& assignCell = multistage->getChild(3);
  ASSERT_EQ(ast::LocationType::Cells, assignCell->getLocationType());

  const auto& assignEdge = multistage->getChild(4);
  ASSERT_EQ(ast::LocationType::Edges, assignEdge->getLocationType());

  const auto& assignVertex = multistage->getChild(5);
  ASSERT_EQ(ast::LocationType::Vertices, assignVertex->getLocationType());
}

TEST_F(TestPassSetStageLocationType, IfStmt) {
  // field(cells) in_cell;
  // var out_var_cell;
  // if(out_var_cell) out_var_cell = in_cell;

  runPass("input/test_set_stage_location_type_if_stmt.sir");

  auto const& multistage = instantiation_->getStencils()[0]->getChild(0);

  const auto& varDeclCell = multistage->getChild(0);
  ASSERT_EQ(ast::LocationType::Cells, varDeclCell->getLocationType());

  const auto& ifCell = multistage->getChild(1);
  ASSERT_EQ(ast::LocationType::Cells, ifCell->getLocationType());
}

// TODO to run this test from IIR, we need serialization support for stencil functions
TEST_F(TestPassSetStageLocationType, FunctionCall) {
  // stencil_function f(field(cells) out) {
  //  out = 2.0;
  // }
  //
  // fields(cells) out_cell;
  // f(out_cells);

  runPass("input/test_set_stage_location_type_function_call.sir");

  auto const& multistage = instantiation_->getStencils()[0]->getChild(0);

  const auto& funCall = multistage->getChild(0);
  ASSERT_EQ(ast::LocationType::Cells, funCall->getLocationType());
}

} // namespace
