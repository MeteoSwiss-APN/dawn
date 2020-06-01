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
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassRemoveScalars.h"
#include "dawn/Support/Logger.h"
#include "dawn/Unittest/ASTConstructionAliases.h"
#include "dawn/Unittest/IIRBuilder.h"
#include "dawn/Unittest/UnittestUtils.h"

#include <gtest/gtest.h>
#include <sstream>

using namespace dawn;
using namespace astgen;

namespace {

bool isVarInDoMethodsAccesses(int varAccessID, const iir::DoMethod& doMethod) {
  for(const auto& stmt : doMethod.getAST().getStatements()) {
    const auto& access = stmt->getData<iir::IIRStmtData>().CallerAccesses;

    if(std::any_of(
           access->getWriteAccesses().begin(), access->getWriteAccesses().end(),
           [&](const auto& pair) { return pair.first == varAccessID; })) { // pair.first = AccessID
      return true;
    }

    if(std::any_of(
           access->getReadAccesses().begin(), access->getReadAccesses().end(),
           [&](const auto& pair) { return pair.first == varAccessID; })) { // pair.first = AccessID
      return true;
    }
  }
  return false;
} // namespace

TEST(TestRemoveScalars, test_unstructured_scalar_01) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_c = b.field("f_c", ast::LocationType::Cells);
  auto varA =
      b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(3.0)}, iir::LocalVariableType::Scalar);

  /// field(cells) f_c;
  /// double varA = 3.0;
  /// f_c = varA;

  auto stencil = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                             b.declareVar(varA), b.stmt(b.assignExpr(b.at(f_c), b.at(varA))))))));

  auto& metadata = stencil->getMetaData();
  int varAID = metadata.getAccessIDFromName("varA");

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  PassRemoveScalars passRemoveScalars(optimizer);
  passRemoveScalars.run(stencil);

  // Check that there is 1 statement left
  ASSERT_EQ(getFirstDoMethod(stencil).getAST().getStatements().size(), 1);

  auto firstStatement = getNthStmt(getFirstDoMethod(stencil), 0);
  // Check that first statement is: f_c = 3.0;
  ASSERT_TRUE(firstStatement->equals(expr(assign(field("f_c"), lit(3.0))).get(),
                                     /*compareData = */ false));
  // Check that variable's metadata is gone
  ASSERT_EQ(metadata.getAccessIDToLocalVariableDataMap().count(varAID), 0);
  // Check that statements' accesses do not contain the variable
  ASSERT_FALSE(isVarInDoMethodsAccesses(varAID, getFirstDoMethod(stencil)));
}

TEST(TestRemoveScalars, test_unstructured_scalar_02) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_c = b.field("f_c", ast::LocationType::Cells);
  auto f_c_out = b.field("f_c_out", ast::LocationType::Cells);
  auto varA =
      b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(3.0)}, iir::LocalVariableType::Scalar);
  auto varB =
      b.localvar("varB", dawn::BuiltinTypeID::Double, {b.lit(5.0)}, iir::LocalVariableType::Scalar);

  /// field(cells) f_c, f_c_out;
  /// double varA = 3.0;
  /// double varB = 5.0;
  /// varA = varA + 1.0;
  /// varB = varB + varA;
  /// f_c = varB;
  /// f_c_out = f_c;

  auto stencil = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(b.doMethod(
              dawn::sir::Interval::Start, dawn::sir::Interval::End, b.declareVar(varA),
              b.declareVar(varB),
              b.stmt(b.assignExpr(b.at(varA), b.binaryExpr(b.at(varA), b.lit(1.0), Op::plus))),
              b.stmt(b.assignExpr(b.at(varB), b.binaryExpr(b.at(varB), b.at(varA), Op::plus))),
              b.stmt(b.assignExpr(b.at(f_c), b.at(varB))),
              b.stmt(b.assignExpr(b.at(f_c_out), b.at(f_c))))))));

  auto& metadata = stencil->getMetaData();
  int varAID = metadata.getAccessIDFromName("varA");
  int varBID = metadata.getAccessIDFromName("varB");

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  PassRemoveScalars passRemoveScalars(optimizer);
  passRemoveScalars.run(stencil);

  // Check that there are 2 statements left
  ASSERT_EQ(getFirstDoMethod(stencil).getAST().getStatements().size(), 2);

  auto firstStatement = getNthStmt(getFirstDoMethod(stencil), 0);
  // Check that first statement is: f_c = 5.0 + (3.0 + 1.0);
  ASSERT_TRUE(firstStatement->equals(
      expr(assign(field("f_c"), binop(lit(5.0), "+", binop(lit(3.0), "+", lit(1.0))))).get(),
      /*compareData = */ false));
  // Check that variables' metadata is gone
  ASSERT_EQ(metadata.getAccessIDToLocalVariableDataMap().count(varAID), 0);
  ASSERT_EQ(metadata.getAccessIDToLocalVariableDataMap().count(varBID), 0);
  // Check that statements' accesses do not contain the variables
  ASSERT_FALSE(isVarInDoMethodsAccesses(varAID, getFirstDoMethod(stencil)));
  ASSERT_FALSE(isVarInDoMethodsAccesses(varBID, getFirstDoMethod(stencil)));
}

TEST(TestRemoveScalars, test_cartesian_scalar_01) {
  using namespace dawn::iir;

  CartesianIIRBuilder b;
  auto f = b.field("f");
  auto varA =
      b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(3.0)}, iir::LocalVariableType::Scalar);
  auto varB =
      b.localvar("varB", dawn::BuiltinTypeID::Double, {b.at(varA)}, iir::LocalVariableType::Scalar);

  /// field_ijk f;
  /// double varA = 3.0;
  /// double varB = varA;
  /// f = varB;

  auto stencil = b.build(
      "generated", b.stencil(b.multistage(
                       dawn::iir::LoopOrderKind::Forward,
                       b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                          b.declareVar(varA), b.declareVar(varB),
                                          b.stmt(b.assignExpr(b.at(f), b.at(varB))))))));

  auto& metadata = stencil->getMetaData();
  int varAID = metadata.getAccessIDFromName("varA");
  int varBID = metadata.getAccessIDFromName("varB");

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  PassRemoveScalars passRemoveScalars(optimizer);
  passRemoveScalars.run(stencil);

  // Check that there is 1 statement left
  ASSERT_EQ(getFirstDoMethod(stencil).getAST().getStatements().size(), 1);

  auto firstStatement = getNthStmt(getFirstDoMethod(stencil), 0);
  // Check that first statement is: f = 3.0;
  ASSERT_TRUE(firstStatement->equals(expr(assign(field("f"), lit(3.0))).get(),
                                     /*compareData = */ false));
  // Check that variables' metadata is gone
  ASSERT_EQ(metadata.getAccessIDToLocalVariableDataMap().count(varAID), 0);
  ASSERT_EQ(metadata.getAccessIDToLocalVariableDataMap().count(varBID), 0);
  // Check that statements' accesses do not contain the variables
  ASSERT_FALSE(isVarInDoMethodsAccesses(varAID, getFirstDoMethod(stencil)));
  ASSERT_FALSE(isVarInDoMethodsAccesses(varBID, getFirstDoMethod(stencil)));
}

TEST(TestRemoveScalars, test_cartesian_scalar_02) {
  using namespace dawn::iir;

  CartesianIIRBuilder b;
  auto f = b.field("f");
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {}, iir::LocalVariableType::Scalar);
  auto varB =
      b.localvar("varB", dawn::BuiltinTypeID::Double, {b.at(varA)}, iir::LocalVariableType::Scalar);

  /// field_ijk f;
  /// double varA;
  /// varA = 3.0;
  /// double varB = varA;
  /// f = varB;

  auto stencil = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                             b.declareVar(varA), b.stmt(b.assignExpr(b.at(varA), b.lit(3.0))),
                             b.declareVar(varB), b.stmt(b.assignExpr(b.at(f), b.at(varB))))))));

  auto& metadata = stencil->getMetaData();
  int varAID = metadata.getAccessIDFromName("varA");
  int varBID = metadata.getAccessIDFromName("varB");

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  PassRemoveScalars passRemoveScalars(optimizer);
  passRemoveScalars.run(stencil);

  // Check that there is 1 statement left
  ASSERT_EQ(getFirstDoMethod(stencil).getAST().getStatements().size(), 1);

  auto firstStatement = getNthStmt(getFirstDoMethod(stencil), 0);
  // Check that first statement is: f = 3.0;
  ASSERT_TRUE(firstStatement->equals(expr(assign(field("f"), lit(3.0))).get(),
                                     /*compareData = */ false));
  // Check that variables' metadata is gone
  ASSERT_EQ(metadata.getAccessIDToLocalVariableDataMap().count(varAID), 0);
  ASSERT_EQ(metadata.getAccessIDToLocalVariableDataMap().count(varBID), 0);
  // Check that statements' accesses do not contain the variables
  ASSERT_FALSE(isVarInDoMethodsAccesses(varAID, getFirstDoMethod(stencil)));
  ASSERT_FALSE(isVarInDoMethodsAccesses(varBID, getFirstDoMethod(stencil)));
}

TEST(TestRemoveScalars, test_global_01) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_c = b.field("f_c", ast::LocationType::Cells);
  auto pi = b.globalvar("pi", 3.14);
  auto varA =
      b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(2.0)}, iir::LocalVariableType::Scalar);

  /// field(cells) f_c;
  /// global double pi = 3.14;
  /// double varA = 2.0;
  /// varA = pi * 2.0;
  /// f_c = varA;

  auto stencil = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(b.doMethod(
              dawn::sir::Interval::Start, dawn::sir::Interval::End, b.declareVar(varA),
              b.stmt(b.assignExpr(b.at(varA), b.binaryExpr(b.at(pi), b.lit(2.0), Op::multiply))),
              b.stmt(b.assignExpr(b.at(f_c), b.at(varA))))))));

  auto& metadata = stencil->getMetaData();
  int varAID = metadata.getAccessIDFromName("varA");

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  PassRemoveScalars passRemoveScalars(optimizer);
  passRemoveScalars.run(stencil);

  // Check that there is 1 statement
  ASSERT_EQ(getFirstDoMethod(stencil).getAST().getStatements().size(), 1);

  auto firstStatement = getNthStmt(getFirstDoMethod(stencil), 0);
  // Check that first statement is: f_c = pi * 2.0;
  ASSERT_TRUE(
      firstStatement->equals(expr(assign(field("f_c"), binop(global("pi"), "*", lit(2.0)))).get(),
                             /*compareData = */ false));
  // Check that variables' metadata is gone
  ASSERT_EQ(metadata.getAccessIDToLocalVariableDataMap().count(varAID), 0);
  // Check that statements' accesses do not contain the variables
  ASSERT_FALSE(isVarInDoMethodsAccesses(varAID, getFirstDoMethod(stencil)));
}

TEST(TestRemoveScalars, test_if_01) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_c = b.field("f_c", ast::LocationType::Cells);
  auto f_c_out = b.field("f_c_out", ast::LocationType::Cells);
  auto varA =
      b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(2.0)}, iir::LocalVariableType::Scalar);

  // field(cells) f_c, f_c_out;
  // double varA = 2.0;
  // if(f_c > 0.0) {
  //   f_c_out = varA;
  // }
  //

  auto stencil = b.build(
      "generated", b.stencil(b.multistage(
                       dawn::iir::LoopOrderKind::Forward,
                       b.stage(b.doMethod(
                           dawn::sir::Interval::Start, dawn::sir::Interval::End, b.declareVar(varA),
                           b.ifStmt(b.binaryExpr(b.at(f_c), b.lit(0.0), Op::greater),
                                    b.block(b.stmt(b.assignExpr(b.at(f_c_out), b.at(varA))))))))));

  auto& metadata = stencil->getMetaData();
  int varAID = metadata.getAccessIDFromName("varA");

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  PassRemoveScalars passRemoveScalars(optimizer);
  passRemoveScalars.run(stencil);

  // Check that there is 1 statement
  ASSERT_EQ(getFirstDoMethod(stencil).getAST().getStatements().size(), 1);

  auto firstStatement = getNthStmt(getFirstDoMethod(stencil), 0);
  // Check that first statement is:
  // if(f_c > 0.0) {
  //   f_c_out = 2.0;
  // }
  ASSERT_TRUE(firstStatement->equals(ifstmt(expr(binop(field("f_c"), ">", lit(0.0))),
                                            block(expr(assign(field("f_c_out"), lit(2.0)))))
                                         .get(),
                                     /*compareData = */ false));
  // Check that variables' metadata is gone
  ASSERT_EQ(metadata.getAccessIDToLocalVariableDataMap().count(varAID), 0);
  // Check that statements' accesses do not contain the variables
  ASSERT_FALSE(isVarInDoMethodsAccesses(varAID, getFirstDoMethod(stencil)));
}

TEST(TestRemoveScalars, test_if_02) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_c = b.field("f_c", ast::LocationType::Cells);
  auto f_c_out = b.field("f_c_out", ast::LocationType::Cells);
  auto varA =
      b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(2.0)}, iir::LocalVariableType::Scalar);

  // field(cells) f_c, f_c_out;
  // double varA = 2.0;
  // if(f_c > varA) {
  //   f_c_out = varA;
  // }
  //

  auto stencil = b.build(
      "generated", b.stencil(b.multistage(
                       dawn::iir::LoopOrderKind::Forward,
                       b.stage(b.doMethod(
                           dawn::sir::Interval::Start, dawn::sir::Interval::End, b.declareVar(varA),
                           b.ifStmt(b.binaryExpr(b.at(f_c), b.at(varA), Op::greater),
                                    b.block(b.stmt(b.assignExpr(b.at(f_c_out), b.at(varA))))))))));

  auto& metadata = stencil->getMetaData();
  int varAID = metadata.getAccessIDFromName("varA");

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  PassRemoveScalars passRemoveScalars(optimizer);
  passRemoveScalars.run(stencil);

  // Check that there is 1 statement
  ASSERT_EQ(getFirstDoMethod(stencil).getAST().getStatements().size(), 1);

  auto firstStatement = getNthStmt(getFirstDoMethod(stencil), 0);
  // Check that first statement is:
  // if(f_c > 2.0) {
  //   f_c_out = 2.0;
  // }
  ASSERT_TRUE(firstStatement->equals(ifstmt(expr(binop(field("f_c"), ">", lit(2.0))),
                                            block(expr(assign(field("f_c_out"), lit(2.0)))))
                                         .get(),
                                     /*compareData = */ false));
  // Check that variables' metadata is gone
  ASSERT_EQ(metadata.getAccessIDToLocalVariableDataMap().count(varAID), 0);
  // Check that statements' accesses do not contain the variables
  ASSERT_FALSE(isVarInDoMethodsAccesses(varAID, getFirstDoMethod(stencil)));
}

TEST(TestRemoveScalars, test_else_01) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_c = b.field("f_c", ast::LocationType::Cells);
  auto f_c_out = b.field("f_c_out", ast::LocationType::Cells);
  auto varA =
      b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(2.0)}, iir::LocalVariableType::Scalar);

  // field(cells) f_c, f_c_out;
  // double varA = 2.0;
  // if(f_c > 0.0) {
  //
  // } else {
  //   f_c_out = varA + 1.0;
  // }
  //

  auto stencil = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(b.doMethod(
              dawn::sir::Interval::Start, dawn::sir::Interval::End, b.declareVar(varA),
              b.ifStmt(b.binaryExpr(b.at(f_c), b.lit(0.0), Op::greater), b.block(),
                       b.block(b.stmt(b.assignExpr(
                           b.at(f_c_out), b.binaryExpr(b.at(varA), b.lit(1.0), Op::plus))))))))));

  auto& metadata = stencil->getMetaData();
  int varAID = metadata.getAccessIDFromName("varA");

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  PassRemoveScalars passRemoveScalars(optimizer);
  passRemoveScalars.run(stencil);

  // Check that there is 1 statement
  ASSERT_EQ(getFirstDoMethod(stencil).getAST().getStatements().size(), 1);

  auto firstStatement = getNthStmt(getFirstDoMethod(stencil), 0);
  // Check that first statement is:
  // if(f_c > 0.0) {
  //
  // } else {
  //   f_c_out = 2.0 + 1.0;
  // }
  ASSERT_TRUE(firstStatement->equals(
      ifstmt(expr(binop(field("f_c"), ">", lit(0.0))), block(),
             block(expr(assign(field("f_c_out"), binop(lit(2.0), "+", lit(1.0))))))
          .get(),
      /*compareData = */ false));

  // Check that variables' metadata is gone
  ASSERT_EQ(metadata.getAccessIDToLocalVariableDataMap().count(varAID), 0);
  // Check that statements' accesses do not contain the variables
  ASSERT_FALSE(isVarInDoMethodsAccesses(varAID, getFirstDoMethod(stencil)));
}

TEST(TestRemoveScalars, warn_compound_assignments) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_e = b.field("f_e", ast::LocationType::Edges);
  auto varA =
      b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(1.0)}, iir::LocalVariableType::Scalar);

  /// field(edges) f_e;
  /// double varA = 1.0;
  /// varA *= 3.0;
  /// f_e = varA;

  auto stencil =
      b.build("generated",
              b.stencil(b.multistage(
                  dawn::iir::LoopOrderKind::Forward,
                  b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                     b.declareVar(varA),
                                     b.stmt(b.assignExpr(b.at(varA), b.lit(3.0), Op::multiply)),
                                     b.stmt(b.assignExpr(b.at(f_e), b.at(varA))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  std::ostringstream output;
  dawn::log::info.stream(output);
  dawn::log::setVerbosity(dawn::log::Level::All);

  // run single pass (PassRemoveScalars) and expect info in output
  PassRemoveScalars passRemoveScalars(optimizer);
  passRemoveScalars.run(stencil);
  ASSERT_NE(output.str().find("Skipping removal of scalar variables."), std::string::npos);
}

TEST(TestRemoveScalars, warn_increment) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_e = b.field("f_e", ast::LocationType::Edges);
  auto varA =
      b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(1.0)}, iir::LocalVariableType::Scalar);

  /// field(edges) f_e;
  /// double varA = 1.0;
  /// varA++;
  /// f_e = varA;

  auto stencil = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                             b.declareVar(varA), b.stmt(b.unaryExpr(b.at(varA), Op::increment)),
                             b.stmt(b.assignExpr(b.at(f_e), b.at(varA))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  std::ostringstream output;
  dawn::log::info.stream(output);
  dawn::log::setVerbosity(dawn::log::Level::All);

  // run single pass (PassRemoveScalars) and expect info in output
  PassRemoveScalars passRemoveScalars(optimizer);
  passRemoveScalars.run(stencil);
  ASSERT_NE(output.str().find("Skipping removal of scalar variables."), std::string::npos);
}

TEST(TestRemoveScalars, warn_condition_adimensional_01) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_e = b.field("f_e", ast::LocationType::Edges);
  auto varA =
      b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(1.0)}, iir::LocalVariableType::Scalar);

  // field(edges) f_e;
  // double varA = 1.0;
  // if(varA == 0.0) {
  //    varA = 4.0;
  // }
  // f_e = varA;

  auto stencil = b.build(
      "generated", b.stencil(b.multistage(
                       dawn::iir::LoopOrderKind::Forward,
                       b.stage(b.doMethod(
                           dawn::sir::Interval::Start, dawn::sir::Interval::End, b.declareVar(varA),
                           b.ifStmt(b.binaryExpr(b.at(varA), b.lit(0.0), Op::equal),
                                    b.block(b.stmt(b.assignExpr(b.at(varA), b.lit(4.0))))),
                           b.stmt(b.assignExpr(b.at(f_e), b.at(varA))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  std::ostringstream output;
  dawn::log::info.stream(output);
  dawn::log::setVerbosity(dawn::log::Level::All);

  // run single pass (PassRemoveScalars) and expect info in output
  PassRemoveScalars passRemoveScalars(optimizer);
  passRemoveScalars.run(stencil);
  ASSERT_NE(output.str().find("Skipping removal of scalar variables."), std::string::npos);
}

TEST(TestRemoveScalars, warn_condition_adimensional_02) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_e = b.field("f_e", ast::LocationType::Edges);
  auto varA =
      b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(1.0)}, iir::LocalVariableType::Scalar);

  // field(edges) f_e;
  // double varA = 1.0;
  // if(1.0 - 1.0) {
  //    varA = 4.0;
  // }
  // f_e = varA;

  auto stencil = b.build(
      "generated", b.stencil(b.multistage(
                       dawn::iir::LoopOrderKind::Forward,
                       b.stage(b.doMethod(
                           dawn::sir::Interval::Start, dawn::sir::Interval::End, b.declareVar(varA),
                           b.ifStmt(b.binaryExpr(b.lit(1.0), b.lit(1.0), Op::minus),
                                    b.block(b.stmt(b.assignExpr(b.at(varA), b.lit(4.0))))),
                           b.stmt(b.assignExpr(b.at(f_e), b.at(varA))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  std::ostringstream output;
  dawn::log::info.stream(output);
  dawn::log::setVerbosity(dawn::log::Level::All);

  // run single pass (PassRemoveScalars) and expect info in output
  PassRemoveScalars passRemoveScalars(optimizer);
  passRemoveScalars.run(stencil);
  ASSERT_NE(output.str().find("Skipping removal of scalar variables."), std::string::npos);
}

TEST(TestRemoveScalars, warn_condition_adimensional_03) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_e = b.field("f_e", ast::LocationType::Edges);
  auto myBool = b.globalvar("myBool", true);
  auto varA =
      b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(1.0)}, iir::LocalVariableType::Scalar);

  // field(edges) f_e;
  // global bool myBool = true;
  // double varA = 1.0;
  // if(myBool) {
  //    varA = 4.0;
  // }
  // f_e = varA;

  auto stencil =
      b.build("generated",
              b.stencil(b.multistage(
                  dawn::iir::LoopOrderKind::Forward,
                  b.stage(b.doMethod(
                      dawn::sir::Interval::Start, dawn::sir::Interval::End, b.declareVar(varA),
                      b.ifStmt(b.at(myBool), b.block(b.stmt(b.assignExpr(b.at(varA), b.lit(4.0))))),
                      b.stmt(b.assignExpr(b.at(f_e), b.at(varA))))))));

  std::ostringstream output;
  dawn::log::info.stream(output);
  dawn::log::setVerbosity(dawn::log::Level::All);

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  // run single pass (PassRemoveScalars) and expect info in output
  PassRemoveScalars passRemoveScalars(optimizer);
  passRemoveScalars.run(stencil);
  ASSERT_NE(output.str().find("Skipping removal of scalar variables."), std::string::npos);
}

} // namespace
