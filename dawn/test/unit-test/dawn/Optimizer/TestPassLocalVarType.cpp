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
#include "dawn/Optimizer/PassLocalVarType.h"
#include "dawn/Unittest/IIRBuilder.h"

#include <gtest/gtest.h>

using namespace dawn;

namespace {

TEST(TestLocalVarType, test_cartesian_01) {
  using namespace dawn::iir;

  CartesianIIRBuilder b;
  auto fIJ = b.field("f_ij", FieldType::ij);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(3.0)});

  /// storage_ij f_ij;
  /// double varA = 3.0;
  /// f_ij = varA;

  auto stencil = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                             b.declareVar(varA), b.stmt(b.assignExpr(b.at(fIJ), b.at(varA))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Cartesian));

  // run single pass (PassLocalVarType)
  PassLocalVarType passLocalVarType(optimizer);
  passLocalVarType.run(stencil);

  int varAID = stencil->getMetaData().getAccessIDFromName("varA");
  // Need to check that varA has been flagged as scalar
  ASSERT_TRUE(stencil->getMetaData().getLocalVariableDataFromAccessID(varAID).isScalar());
}

TEST(TestLocalVarType, test_cartesian_02) {
  using namespace dawn::iir;

  CartesianIIRBuilder b;
  auto fIJ = b.field("f_ij", FieldType::ij);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(3.0)});

  /// storage_ij f_ij;
  /// double varA = 3.0;
  /// varA = f_ij;
  /// f_ij = varA;

  auto stencil = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                             b.declareVar(varA), b.stmt(b.assignExpr(b.at(varA), b.at(fIJ))),
                             b.stmt(b.assignExpr(b.at(fIJ), b.at(varA))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Cartesian));

  // run single pass (PassLocalVarType)
  PassLocalVarType passLocalVarType(optimizer);
  passLocalVarType.run(stencil);

  int varAID = stencil->getMetaData().getAccessIDFromName("varA");
  // Need to check that varA has been flagged as OnIJ
  ASSERT_EQ(stencil->getMetaData().getLocalVariableDataFromAccessID(varAID).getType(),
            iir::LocalVariableType::OnIJ);
}

TEST(TestLocalVarType, test_cartesian_propagation_01) {
  using namespace dawn::iir;

  CartesianIIRBuilder b;
  auto fIJ = b.field("f_ij", FieldType::ij);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(3.0)});
  auto varB = b.localvar("varB", dawn::BuiltinTypeID::Double, {b.at(varA)});

  /// storage_ij f_ij;
  /// double varA = 3.0;
  /// double varB = varA;
  /// f_ij = varB;

  auto stencil = b.build(
      "generated", b.stencil(b.multistage(
                       dawn::iir::LoopOrderKind::Forward,
                       b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                          b.declareVar(varA), b.declareVar(varB),
                                          b.stmt(b.assignExpr(b.at(fIJ), b.at(varB))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Cartesian));

  // run single pass (PassLocalVarType)
  PassLocalVarType passLocalVarType(optimizer);
  passLocalVarType.run(stencil);

  int varAID = stencil->getMetaData().getAccessIDFromName("varA");
  int varBID = stencil->getMetaData().getAccessIDFromName("varB");
  // Need to check that varA has been flagged as scalar
  ASSERT_TRUE(stencil->getMetaData().getLocalVariableDataFromAccessID(varAID).isScalar());
  // Need to check that varB has been flagged as scalar
  ASSERT_TRUE(stencil->getMetaData().getLocalVariableDataFromAccessID(varBID).isScalar());
}

TEST(TestLocalVarType, test_cartesian_propagation_02) {
  using namespace dawn::iir;

  CartesianIIRBuilder b;
  auto fIJ = b.field("f_ij", FieldType::ij);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(3.0)});
  auto varB = b.localvar("varB", dawn::BuiltinTypeID::Double, {b.at(varA)});

  /// storage_ij f_ij;
  /// double varA = 3.0;
  /// double varB = varA;
  /// varA = f_ij;
  /// f_ij = varB;

  auto stencil = b.build(
      "generated", b.stencil(b.multistage(
                       dawn::iir::LoopOrderKind::Forward,
                       b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                          b.declareVar(varA), b.declareVar(varB),
                                          b.stmt(b.assignExpr(b.at(varA), b.at(fIJ))),
                                          b.stmt(b.assignExpr(b.at(fIJ), b.at(varB))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Cartesian));

  // run single pass (PassLocalVarType)
  PassLocalVarType passLocalVarType(optimizer);
  passLocalVarType.run(stencil);

  int varAID = stencil->getMetaData().getAccessIDFromName("varA");
  int varBID = stencil->getMetaData().getAccessIDFromName("varB");
  // Need to check that varA has been flagged as IJ
  ASSERT_EQ(stencil->getMetaData().getLocalVariableDataFromAccessID(varAID).getType(),
            iir::LocalVariableType::OnIJ);
  // Need to check that varB has been flagged as IJ
  ASSERT_EQ(stencil->getMetaData().getLocalVariableDataFromAccessID(varBID).getType(),
            iir::LocalVariableType::OnIJ);
}

TEST(TestLocalVarType, test_cartesian_propagation_03) {
  using namespace dawn::iir;

  CartesianIIRBuilder b;
  auto fIJ = b.field("f_ij", FieldType::ij);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(3.0)});
  auto varB = b.localvar("varB", dawn::BuiltinTypeID::Double, {b.lit(1.0)});
  auto varC = b.localvar("varC", dawn::BuiltinTypeID::Double, {b.lit(2.0)});
  auto varD = b.localvar("varD", dawn::BuiltinTypeID::Double, {b.at(varC)});

  /// storage_ij f_ij;
  /// double varA = 3.0;
  /// double varB = 1.0;
  /// double varC = 2.0;
  /// double varD = varC;
  /// varB = varA;
  /// varD = varB;
  /// f_ij = varD;

  auto stencil = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                             b.declareVar(varA), b.declareVar(varB), b.declareVar(varC),
                             b.declareVar(varD), b.stmt(b.assignExpr(b.at(varB), b.at(varA))),
                             b.stmt(b.assignExpr(b.at(varD), b.at(varB))),
                             b.stmt(b.assignExpr(b.at(fIJ), b.at(varD))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Cartesian));

  // run single pass (PassLocalVarType)
  PassLocalVarType passLocalVarType(optimizer);
  passLocalVarType.run(stencil);

  int varAID = stencil->getMetaData().getAccessIDFromName("varA");
  int varBID = stencil->getMetaData().getAccessIDFromName("varB");
  int varCID = stencil->getMetaData().getAccessIDFromName("varC");
  int varDID = stencil->getMetaData().getAccessIDFromName("varD");
  // Need to check that varA has been flagged as scalar
  ASSERT_TRUE(stencil->getMetaData().getLocalVariableDataFromAccessID(varAID).isScalar());
  // Need to check that varB has been flagged as scalar
  ASSERT_TRUE(stencil->getMetaData().getLocalVariableDataFromAccessID(varBID).isScalar());
  // Need to check that varC has been flagged as scalar
  ASSERT_TRUE(stencil->getMetaData().getLocalVariableDataFromAccessID(varCID).isScalar());
  // Need to check that varD has been flagged as scalar
  ASSERT_TRUE(stencil->getMetaData().getLocalVariableDataFromAccessID(varDID).isScalar());
}

TEST(TestLocalVarType, test_cartesian_propagation_04) {
  using namespace dawn::iir;

  CartesianIIRBuilder b;
  auto fIJ = b.field("f_ij", FieldType::ij);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(3.0)});
  auto varB = b.localvar("varB", dawn::BuiltinTypeID::Double, {b.lit(1.0)});
  auto varC = b.localvar("varC", dawn::BuiltinTypeID::Double, {b.lit(2.0)});
  auto varD = b.localvar("varD", dawn::BuiltinTypeID::Double, {b.at(varC)});

  /// storage_ij f_ij;
  /// double varA = 3.0;
  /// double varB = 1.0;
  /// double varC = 2.0;
  /// double varD = varC;
  /// varB = varA;
  /// varD = varB;
  /// varB = f_ij;
  /// f_ij = varD;

  auto stencil = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                             b.declareVar(varA), b.declareVar(varB), b.declareVar(varC),
                             b.declareVar(varD), b.stmt(b.assignExpr(b.at(varB), b.at(varA))),
                             b.stmt(b.assignExpr(b.at(varD), b.at(varB))),
                             b.stmt(b.assignExpr(b.at(varB), b.at(fIJ))),
                             b.stmt(b.assignExpr(b.at(fIJ), b.at(varD))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Cartesian));

  // run single pass (PassLocalVarType)
  PassLocalVarType passLocalVarType(optimizer);
  passLocalVarType.run(stencil);

  int varAID = stencil->getMetaData().getAccessIDFromName("varA");
  int varBID = stencil->getMetaData().getAccessIDFromName("varB");
  int varCID = stencil->getMetaData().getAccessIDFromName("varC");
  int varDID = stencil->getMetaData().getAccessIDFromName("varD");
  // Need to check that varA has been flagged as scalar
  ASSERT_TRUE(stencil->getMetaData().getLocalVariableDataFromAccessID(varAID).isScalar());
  // Need to check that varB has been flagged as OnIJ
  ASSERT_EQ(stencil->getMetaData().getLocalVariableDataFromAccessID(varBID).getType(),
            iir::LocalVariableType::OnIJ);
  // Need to check that varC has been flagged as scalar
  ASSERT_TRUE(stencil->getMetaData().getLocalVariableDataFromAccessID(varCID).isScalar());
  // Need to check that varD has been flagged as OnIJ
  ASSERT_EQ(stencil->getMetaData().getLocalVariableDataFromAccessID(varDID).getType(),
            iir::LocalVariableType::OnIJ);
}

TEST(TestLocalVarType, test_unstructured_propagation_01) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_c = b.field("f_c", ast::LocationType::Cells);
  auto f_e = b.field("f_e", ast::LocationType::Edges);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(3.0)});
  auto varB = b.localvar("varB", dawn::BuiltinTypeID::Double, {b.lit(2.0)});
  auto varC = b.localvar("varC", dawn::BuiltinTypeID::Double, {b.at(f_e)});
  auto varD = b.localvar("varD", dawn::BuiltinTypeID::Double, {b.lit(3.0)});

  /// field(cells) f_c;
  /// field(edges) f_e;
  /// double varA = 3.0;
  /// double varB = 2.0;
  /// double varC = f_e;
  /// double varD = 3.0;
  /// varA = varB;
  /// varB = varD;
  /// varD = f_c;

  auto stencil = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                             b.declareVar(varA), b.declareVar(varB), b.declareVar(varC),
                             b.declareVar(varD), b.stmt(b.assignExpr(b.at(varA), b.at(varB))),
                             b.stmt(b.assignExpr(b.at(varB), b.at(varD))),
                             b.stmt(b.assignExpr(b.at(varD), b.at(f_c))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  // run single pass (PassLocalVarType)
  PassLocalVarType passLocalVarType(optimizer);
  passLocalVarType.run(stencil);

  int varAID = stencil->getMetaData().getAccessIDFromName("varA");
  int varBID = stencil->getMetaData().getAccessIDFromName("varB");
  int varCID = stencil->getMetaData().getAccessIDFromName("varC");
  int varDID = stencil->getMetaData().getAccessIDFromName("varD");
  // Need to check that varA has been flagged as OnCells
  ASSERT_EQ(stencil->getMetaData().getLocalVariableDataFromAccessID(varAID).getLocationType(),
            ast::LocationType::Cells);
  // Need to check that varB has been flagged as OnCells
  ASSERT_EQ(stencil->getMetaData().getLocalVariableDataFromAccessID(varBID).getLocationType(),
            ast::LocationType::Cells);
  // Need to check that varC has been flagged as OnEdges
  ASSERT_EQ(stencil->getMetaData().getLocalVariableDataFromAccessID(varCID).getLocationType(),
            ast::LocationType::Edges);
  // Need to check that varD has been flagged as OnCells
  ASSERT_EQ(stencil->getMetaData().getLocalVariableDataFromAccessID(varDID).getLocationType(),
            ast::LocationType::Cells);
}

TEST(TestLocalVarType, test_unstructured_propagation_02) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_e = b.field("f_e", ast::LocationType::Edges);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.at(f_e)});
  auto varB = b.localvar("varB", dawn::BuiltinTypeID::Double, {b.lit(2.0)});
  auto varC = b.localvar("varC", dawn::BuiltinTypeID::Double, {b.lit(3.0)});

  /// field(edges) f_e;
  /// double varA = f_e;
  /// double varB = 2.0;
  /// double varC = 3.0;
  /// varA = varB;
  /// varB = varC;

  auto stencil =
      b.build("generated",
              b.stencil(b.multistage(
                  dawn::iir::LoopOrderKind::Forward,
                  b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                     b.declareVar(varA), b.declareVar(varB), b.declareVar(varC),
                                     b.stmt(b.assignExpr(b.at(varA), b.at(varB))),
                                     b.stmt(b.assignExpr(b.at(varB), b.at(varC))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  // run single pass (PassLocalVarType)
  PassLocalVarType passLocalVarType(optimizer);
  passLocalVarType.run(stencil);

  int varAID = stencil->getMetaData().getAccessIDFromName("varA");
  int varBID = stencil->getMetaData().getAccessIDFromName("varB");
  int varCID = stencil->getMetaData().getAccessIDFromName("varC");
  // Need to check that varA has been flagged as OnEdges
  ASSERT_EQ(stencil->getMetaData().getLocalVariableDataFromAccessID(varAID).getLocationType(),
            ast::LocationType::Edges);
  // Need to check that varB has been flagged as scalar
  ASSERT_TRUE(stencil->getMetaData().getLocalVariableDataFromAccessID(varBID).isScalar());
  // Need to check that varC has been flagged as scalar
  ASSERT_TRUE(stencil->getMetaData().getLocalVariableDataFromAccessID(varCID).isScalar());
}

TEST(TestLocalVarType, test_unstructured_if_condition_01) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_e = b.field("f_e", ast::LocationType::Edges);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(2.0)});
  auto varB = b.localvar("varB", dawn::BuiltinTypeID::Double, {b.lit(1.0)});
  auto varC = b.localvar("varC", dawn::BuiltinTypeID::Double, {b.lit(0.0)});

  /// field(edges) f_e;
  /// double varA = 2.0;
  /// double varB = 1.0;
  /// if(f_e > 0.0) {
  ///    varA = 3.0;
  /// } else {
  ///    varB = 5.0;
  ///    double varC = 0.0;
  /// }

  auto stencil =
      b.build("generated",
              b.stencil(b.multistage(
                  dawn::iir::LoopOrderKind::Forward,
                  b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                     b.declareVar(varA), b.declareVar(varB),
                                     b.ifStmt(b.binaryExpr(b.at(f_e), b.lit(0.0), Op::greater),
                                              b.block(b.stmt(b.assignExpr(b.at(varA), b.lit(3.0)))),
                                              b.block(b.stmt(b.assignExpr(b.at(varB), b.lit(5.0))),
                                                      b.declareVar(varC))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  // run single pass (PassLocalVarType)
  PassLocalVarType passLocalVarType(optimizer);
  passLocalVarType.run(stencil);

  int varAID = stencil->getMetaData().getAccessIDFromName("varA");
  int varBID = stencil->getMetaData().getAccessIDFromName("varB");
  int varCID = stencil->getMetaData().getAccessIDFromName("varC");
  // Need to check that varA has been flagged as OnEdges
  ASSERT_EQ(stencil->getMetaData().getLocalVariableDataFromAccessID(varAID).getLocationType(),
            ast::LocationType::Edges);
  // Need to check that varB has been flagged as OnEdges
  ASSERT_EQ(stencil->getMetaData().getLocalVariableDataFromAccessID(varBID).getLocationType(),
            ast::LocationType::Edges);
  // Need to check that varC has been flagged as OnEdges
  ASSERT_EQ(stencil->getMetaData().getLocalVariableDataFromAccessID(varCID).getLocationType(),
            ast::LocationType::Edges);
}

TEST(TestLocalVarType, test_unstructured_if_condition_02) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_e = b.field("f_e", ast::LocationType::Edges);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.at(f_e)});
  auto varB = b.localvar("varB", dawn::BuiltinTypeID::Double, {b.at(f_e)});
  auto varC = b.localvar("varC", dawn::BuiltinTypeID::Double, {b.at(f_e)});

  /// field(edges) f_e;
  /// double varA = f_e;
  /// double varB = f_e;
  /// if(f_e > 0.0) {
  ///    varA = 3.0;
  /// } else {
  ///    varB = 5.0;
  ///    double varC = f_e;
  /// }

  auto stencil =
      b.build("generated",
              b.stencil(b.multistage(
                  dawn::iir::LoopOrderKind::Forward,
                  b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                     b.declareVar(varA), b.declareVar(varB),
                                     b.ifStmt(b.binaryExpr(b.at(f_e), b.lit(0.0), Op::greater),
                                              b.block(b.stmt(b.assignExpr(b.at(varA), b.lit(3.0)))),
                                              b.block(b.stmt(b.assignExpr(b.at(varB), b.lit(5.0))),
                                                      b.declareVar(varC))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  // run single pass (PassLocalVarType)
  PassLocalVarType passLocalVarType(optimizer);
  passLocalVarType.run(stencil);

  int varAID = stencil->getMetaData().getAccessIDFromName("varA");
  int varBID = stencil->getMetaData().getAccessIDFromName("varB");
  int varCID = stencil->getMetaData().getAccessIDFromName("varC");
  // Need to check that varA has been flagged as OnEdges
  ASSERT_EQ(stencil->getMetaData().getLocalVariableDataFromAccessID(varAID).getLocationType(),
            ast::LocationType::Edges);
  // Need to check that varB has been flagged as OnEdges
  ASSERT_EQ(stencil->getMetaData().getLocalVariableDataFromAccessID(varBID).getLocationType(),
            ast::LocationType::Edges);
  // Need to check that varC has been flagged as OnEdges
  ASSERT_EQ(stencil->getMetaData().getLocalVariableDataFromAccessID(varCID).getLocationType(),
            ast::LocationType::Edges);
}

TEST(TestLocalVarType, test_unstructured_if_condition_03) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_e = b.field("f_e", ast::LocationType::Edges);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(1.0)});
  auto varB = b.localvar("varB", dawn::BuiltinTypeID::Double, {b.lit(2.0)});

  /// field(edges) f_e;
  /// double varA = 1.0;
  /// double varB = 2.0;
  /// if(varA > 0.0) {
  ///    varB = 3.0;
  /// }
  /// varA = f_e;

  auto stencil = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                             b.declareVar(varA), b.declareVar(varB),
                             b.ifStmt(b.binaryExpr(b.at(varA), b.lit(0.0), Op::greater),
                                      b.block(b.stmt(b.assignExpr(b.at(varB), b.lit(3.0))))),
                             b.stmt(b.assignExpr(b.at(varA), b.at(f_e))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  // run single pass (PassLocalVarType)
  PassLocalVarType passLocalVarType(optimizer);
  passLocalVarType.run(stencil);

  int varAID = stencil->getMetaData().getAccessIDFromName("varA");
  int varBID = stencil->getMetaData().getAccessIDFromName("varB");
  // Need to check that varA has been flagged as OnEdges
  ASSERT_EQ(stencil->getMetaData().getLocalVariableDataFromAccessID(varAID).getLocationType(),
            ast::LocationType::Edges);
  // Need to check that varB has been flagged as OnEdges
  ASSERT_EQ(stencil->getMetaData().getLocalVariableDataFromAccessID(varBID).getLocationType(),
            ast::LocationType::Edges);
}

TEST(TestLocalVarType, test_unstructured_nested_if_01) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_e = b.field("f_e", ast::LocationType::Edges);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(1.0)});
  auto varB = b.localvar("varB", dawn::BuiltinTypeID::Double, {b.lit(2.0)});

  /// field(edges) f_e;
  /// double varA = 1.0;
  /// double varB = 2.0;
  /// if(f_e > 0.0) {
  ///    if(varB > 0.0) {
  ///       varA = 5.0;
  ///    }
  /// }
  /// f_e = varA;

  auto stencil = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(b.doMethod(
              dawn::sir::Interval::Start, dawn::sir::Interval::End, b.declareVar(varA),
              b.declareVar(varB),
              b.ifStmt(b.binaryExpr(b.at(f_e), b.lit(0.0), Op::greater),
                       b.block(b.ifStmt(b.binaryExpr(b.at(varB), b.lit(0.0), Op::greater),
                                        b.block(b.stmt(b.assignExpr(b.at(varA), b.lit(5.0))))))),
              b.stmt(b.assignExpr(b.at(f_e), b.at(varA))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  // run single pass (PassLocalVarType)
  PassLocalVarType passLocalVarType(optimizer);
  passLocalVarType.run(stencil);

  int varAID = stencil->getMetaData().getAccessIDFromName("varA");
  int varBID = stencil->getMetaData().getAccessIDFromName("varB");
  // Need to check that varA has been flagged as OnEdges
  ASSERT_EQ(stencil->getMetaData().getLocalVariableDataFromAccessID(varAID).getLocationType(),
            ast::LocationType::Edges);
  // Need to check that varB has been flagged as scalar
  ASSERT_TRUE(stencil->getMetaData().getLocalVariableDataFromAccessID(varBID).isScalar());
}

TEST(TestLocalVarType, test_unstructured_nested_if_02) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_e = b.field("f_e", ast::LocationType::Edges);
  auto g = b.globalvar("g", 1.0);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(1.0)});
  auto varB = b.localvar("varB", dawn::BuiltinTypeID::Double, {b.lit(2.0)});

  /// field(edges) f_e;
  /// global double g = 1.0;
  /// double varA = 1.0;
  /// double varB = 2.0;
  /// if(g > 0.0) {
  ///    if(varB > 0.0) {
  ///       varA = 5.0;
  ///    }
  /// }
  /// varB = f_e;

  auto stencil = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(b.doMethod(
              dawn::sir::Interval::Start, dawn::sir::Interval::End, b.declareVar(varA),
              b.declareVar(varB),
              b.ifStmt(b.binaryExpr(b.at(g), b.lit(0.0), Op::greater),
                       b.block(b.ifStmt(b.binaryExpr(b.at(varB), b.lit(0.0), Op::greater),
                                        b.block(b.stmt(b.assignExpr(b.at(varA), b.lit(5.0))))))),
              b.stmt(b.assignExpr(b.at(varB), b.at(f_e))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  // run single pass (PassLocalVarType)
  PassLocalVarType passLocalVarType(optimizer);
  passLocalVarType.run(stencil);

  int varAID = stencil->getMetaData().getAccessIDFromName("varA");
  int varBID = stencil->getMetaData().getAccessIDFromName("varB");
  // Need to check that varA has been flagged as OnEdges
  ASSERT_EQ(stencil->getMetaData().getLocalVariableDataFromAccessID(varAID).getLocationType(),
            ast::LocationType::Edges);
  // Need to check that varB has been flagged as scalar
  ASSERT_EQ(stencil->getMetaData().getLocalVariableDataFromAccessID(varBID).getLocationType(),
            ast::LocationType::Edges);
}

TEST(TestLocalVarType, test_unstructured_reduction_01) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_e = b.field("f_e", ast::LocationType::Edges);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(1.0)});

  /// field(edges) f_e;
  /// double varA = 1.0;
  /// varA = reduceEdgesToCells(op = +, init = 0.0, rhs = f_e);

  auto stencil =
      b.build("generated",
              b.stencil(b.multistage(
                  dawn::iir::LoopOrderKind::Forward,
                  b.stage(b.doMethod(
                      dawn::sir::Interval::Start, dawn::sir::Interval::End, b.declareVar(varA),
                      b.stmt(b.assignExpr(
                          b.at(varA), b.reduceOverNeighborExpr(Op::plus, b.at(f_e), b.lit(0.0),
                                                               {ast::LocationType::Cells,
                                                                ast::LocationType::Edges}))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  // run single pass (PassLocalVarType)
  PassLocalVarType passLocalVarType(optimizer);
  passLocalVarType.run(stencil);

  int varAID = stencil->getMetaData().getAccessIDFromName("varA");
  // Need to check that varA has been flagged as OnCells
  ASSERT_EQ(stencil->getMetaData().getLocalVariableDataFromAccessID(varAID).getLocationType(),
            ast::LocationType::Cells);
}

TEST(TestLocalVarType, test_throw_unstructured_01) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_c = b.field("f_c", ast::LocationType::Cells);
  auto f_e = b.field("f_e", ast::LocationType::Edges);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.at(f_c)});

  /// field(cells) f_c;
  /// field(edges) f_e;
  /// double varA = f_c;
  /// varA = f_e;

  auto stencil = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                             b.declareVar(varA), b.stmt(b.assignExpr(b.at(varA), b.at(f_e))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  // run single pass (PassLocalVarType) and expect exception to be thrown
  PassLocalVarType passLocalVarType(optimizer);
  EXPECT_THROW(passLocalVarType.run(stencil), std::runtime_error);
}

TEST(TestLocalVarType, test_throw_unstructured_02) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_c = b.field("f_c", ast::LocationType::Cells);
  auto f_e = b.field("f_e", ast::LocationType::Edges);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(3.0)});
  auto varB = b.localvar("varB", dawn::BuiltinTypeID::Double, {b.lit(2.0)});

  /// field(cells) f_c;
  /// field(edges) f_e;
  /// double varA = 3.0;
  /// double varB = 2.0;
  /// varB = f_e + varA;
  /// varA = f_c;

  auto stencil = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                             b.declareVar(varA), b.declareVar(varB),
                             b.stmt(b.assignExpr(b.at(varB), b.binaryExpr(b.at(f_e), b.at(varA)))),
                             b.stmt(b.assignExpr(b.at(varA), b.at(f_c))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  // run single pass (PassLocalVarType) and expect exception to be thrown
  PassLocalVarType passLocalVarType(optimizer);
  EXPECT_THROW(passLocalVarType.run(stencil), std::runtime_error);
}

TEST(TestLocalVarType, test_throw_unstructured_03) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_c = b.field("f_c", ast::LocationType::Cells);
  auto f_e = b.field("f_e", ast::LocationType::Edges);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(3.0)});
  auto varB = b.localvar("varB", dawn::BuiltinTypeID::Double, {b.lit(2.0)});
  auto varC = b.localvar("varC", dawn::BuiltinTypeID::Double, {b.lit(1.0)});

  /// field(cells) f_c;
  /// field(edges) f_e;
  /// double varA = 3.0;
  /// double varB = 2.0;
  /// double varC = 1.0;
  /// varB = varA;
  /// varC = varB;
  /// varA = f_c;
  /// varC = f_e;

  auto stencil =
      b.build("generated",
              b.stencil(b.multistage(
                  dawn::iir::LoopOrderKind::Forward,
                  b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                     b.declareVar(varA), b.declareVar(varB), b.declareVar(varC),
                                     b.stmt(b.assignExpr(b.at(varB), b.at(varA))),
                                     b.stmt(b.assignExpr(b.at(varC), b.at(varB))),
                                     b.stmt(b.assignExpr(b.at(varA), b.at(f_c))),
                                     b.stmt(b.assignExpr(b.at(varC), b.at(f_e))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  // run single pass (PassLocalVarType) and expect exception to be thrown
  PassLocalVarType passLocalVarType(optimizer);
  EXPECT_THROW(passLocalVarType.run(stencil), std::runtime_error);
}

TEST(TestLocalVarType, test_throw_unstructured_04) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_c = b.field("f_c", ast::LocationType::Cells);
  auto f_e = b.field("f_e", ast::LocationType::Edges);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.at(f_e)});

  /// field(cells) f_c;
  /// field(edges) f_e;
  /// double varA = f_e;
  /// if(f_c > 0.0) {
  ///    varA = 1.0;
  /// }

  auto stencil = b.build(
      "generated", b.stencil(b.multistage(
                       dawn::iir::LoopOrderKind::Forward,
                       b.stage(b.doMethod(
                           dawn::sir::Interval::Start, dawn::sir::Interval::End, b.declareVar(varA),
                           b.ifStmt(b.binaryExpr(b.at(f_c), b.lit(0.0), Op::greater),
                                    b.block(b.stmt(b.assignExpr(b.at(varA), b.lit(1.0))))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  // run single pass (PassLocalVarType) and expect exception to be thrown
  PassLocalVarType passLocalVarType(optimizer);
  EXPECT_THROW(passLocalVarType.run(stencil), std::runtime_error);
}

TEST(TestLocalVarType, test_throw_unstructured_05) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_c = b.field("f_c", ast::LocationType::Cells);
  auto f_e = b.field("f_e", ast::LocationType::Edges);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(2.0)});
  auto varB = b.localvar("varB", dawn::BuiltinTypeID::Double, {b.lit(1.0)});

  /// field(cells) f_c;
  /// field(edges) f_e;
  /// double varA = 2.0;
  /// double varB = 1.0;
  /// if(varA > 0.0) {
  ///    varB = f_c;
  /// }
  /// varA = f_e;

  auto stencil =
      b.build("generated",
              b.stencil(b.multistage(
                  dawn::iir::LoopOrderKind::Forward,
                  b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                     b.declareVar(varA), b.declareVar(varB),
                                     b.ifStmt(b.binaryExpr(b.at(varA), b.lit(0.0), Op::greater),
                                              b.block(b.stmt(b.assignExpr(b.at(varB), b.at(f_c))))),
                                     b.stmt(b.assignExpr(b.at(varA), b.at(f_e))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  // run single pass (PassLocalVarType) and expect exception to be thrown
  PassLocalVarType passLocalVarType(optimizer);
  EXPECT_THROW(passLocalVarType.run(stencil), std::runtime_error);
}

TEST(TestLocalVarType, test_throw_unstructured_06) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_e = b.field("f_e", ast::LocationType::Edges);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(1.0)});

  /// field(edges) f_e;
  /// double varA = 1.0;
  /// varA = reduceEdgesToCells(op = +, init = 0.0, rhs = f_e);
  /// varA = f_e;

  auto stencil = b.build(
      "generated", b.stencil(b.multistage(
                       dawn::iir::LoopOrderKind::Forward,
                       b.stage(b.doMethod(
                           dawn::sir::Interval::Start, dawn::sir::Interval::End, b.declareVar(varA),
                           b.stmt(b.assignExpr(
                               b.at(varA), b.reduceOverNeighborExpr(Op::plus, b.at(f_e), b.lit(0.0),
                                                                    {ast::LocationType::Cells,
                                                                     ast::LocationType::Edges}))),
                           b.stmt(b.assignExpr(b.at(varA), b.at(f_e))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  // run single pass (PassLocalVarType) and expect exception to be thrown
  PassLocalVarType passLocalVarType(optimizer);
  EXPECT_THROW(passLocalVarType.run(stencil), std::runtime_error);
}

TEST(TestLocalVarType, test_throw_nested_if_01) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_c = b.field("f_c", ast::LocationType::Cells);
  auto f_e = b.field("f_e", ast::LocationType::Edges);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(2.0)});

  /// field(cells) f_c;
  /// field(edges) f_e;
  /// double varA = 2.0;
  /// if(f_e > 0.0) {
  ///    if(f_c > 0.0) {
  ///       varA = 1.0;
  ///    }
  /// }

  auto stencil = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(b.doMethod(
              dawn::sir::Interval::Start, dawn::sir::Interval::End, b.declareVar(varA),
              b.ifStmt(
                  b.binaryExpr(b.at(f_e), b.lit(0.0), Op::greater),
                  b.block(b.ifStmt(b.binaryExpr(b.at(f_c), b.lit(0.0), Op::greater),
                                   b.block(b.stmt(b.assignExpr(b.at(varA), b.lit(1.0))))))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  // run single pass (PassLocalVarType) and expect exception to be thrown
  PassLocalVarType passLocalVarType(optimizer);
  EXPECT_THROW(passLocalVarType.run(stencil), std::runtime_error);
}

TEST(TestLocalVarType, test_throw_nested_if_02) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_c = b.field("f_c", ast::LocationType::Cells);
  auto f_e = b.field("f_e", ast::LocationType::Edges);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(2.0)});
  auto varB = b.localvar("varB", dawn::BuiltinTypeID::Double, {b.at(f_c)});
  auto varC = b.localvar("varC", dawn::BuiltinTypeID::Double, {b.lit(1.0)});

  /// field(cells) f_c;
  /// field(edges) f_e;
  /// double varA = 2.0;
  /// double varB = f_c;
  /// double varC = 1.0;
  /// if(varA > 0.0) {
  ///    if(varB > 0.0) {
  ///       varC = 2.0;
  ///    }
  /// }
  /// varA = f_e;

  auto stencil = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(b.doMethod(
              dawn::sir::Interval::Start, dawn::sir::Interval::End, b.declareVar(varA),
              b.declareVar(varB), b.declareVar(varC),
              b.ifStmt(b.binaryExpr(b.at(varA), b.lit(0.0), Op::greater),
                       b.block(b.ifStmt(b.binaryExpr(b.at(varB), b.lit(0.0), Op::greater),
                                        b.block(b.stmt(b.assignExpr(b.at(varC), b.lit(2.0))))))),
              b.stmt(b.assignExpr(b.at(varA), b.at(f_e))))))));

  OptimizerContext::OptimizerContextOptions optimizerOptions;
  OptimizerContext optimizer(optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  // run single pass (PassLocalVarType) and expect exception to be thrown
  PassLocalVarType passLocalVarType(optimizer);
  EXPECT_THROW(passLocalVarType.run(stencil), std::runtime_error);
}

} // namespace
