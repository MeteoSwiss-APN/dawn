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

  auto stencilInstantiationContext = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                             b.declareVar(varA), b.stmt(b.assignExpr(b.at(fIJ), b.at(varA))))))));
  auto stencil = stencilInstantiationContext.at("generated");

  OptimizerContext::OptimizerContextOptions optimizerOptions;

  DawnCompiler compiler;
  OptimizerContext optimizer(compiler.getDiagnostics(), optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Cartesian));

  // run single pass (PassLocalVarType)
  PassLocalVarType passLocalVarType(optimizer);
  passLocalVarType.run(stencil);

  int varAID = stencil->getMetaData().getAccessIDFromName("varA");
  ASSERT_TRUE(stencil->getMetaData().getLocalVariableDataFromAccessID(varAID).isTypeSet());
  // Need to check that varA has been flagged as scalar
  ASSERT_TRUE(stencil->getMetaData().getLocalVariableDataFromAccessID(varAID).isScalar());
}

TEST(TestLocalVarType, test_cartesian_02) {
  using namespace dawn::iir;

  CartesianIIRBuilder b;
  auto fIJ = b.field("f_ij", FieldType::ij);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(3.0)});
  auto varB = b.localvar("varB", dawn::BuiltinTypeID::Double, {b.at(varA)});

  /// storage_ij f_ij;
  /// double varA = 3.0;
  /// double varB = varA;
  /// f_ij = varB;

  auto stencilInstantiationContext = b.build(
      "generated", b.stencil(b.multistage(
                       dawn::iir::LoopOrderKind::Forward,
                       b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                          b.declareVar(varA), b.declareVar(varB),
                                          b.stmt(b.assignExpr(b.at(fIJ), b.at(varB))))))));
  auto stencil = stencilInstantiationContext.at("generated");

  OptimizerContext::OptimizerContextOptions optimizerOptions;

  DawnCompiler compiler;
  OptimizerContext optimizer(compiler.getDiagnostics(), optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Cartesian));

  // run single pass (PassLocalVarType)
  PassLocalVarType passLocalVarType(optimizer);
  passLocalVarType.run(stencil);

  int varAID = stencil->getMetaData().getAccessIDFromName("varA");
  int varBID = stencil->getMetaData().getAccessIDFromName("varB");
  ASSERT_TRUE(stencil->getMetaData().getLocalVariableDataFromAccessID(varAID).isTypeSet());
  // Need to check that varA has been flagged as scalar
  ASSERT_TRUE(stencil->getMetaData().getLocalVariableDataFromAccessID(varAID).isScalar());
  // Need to check that varB has been flagged as scalar
  ASSERT_TRUE(stencil->getMetaData().getLocalVariableDataFromAccessID(varBID).isScalar());
}

TEST(TestLocalVarType, test_cartesian_03) {
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

  auto stencilInstantiationContext = b.build(
      "generated", b.stencil(b.multistage(
                       dawn::iir::LoopOrderKind::Forward,
                       b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                          b.declareVar(varA), b.declareVar(varB),
                                          b.stmt(b.assignExpr(b.at(varA), b.at(fIJ))),
                                          b.stmt(b.assignExpr(b.at(fIJ), b.at(varB))))))));
  auto stencil = stencilInstantiationContext.at("generated");

  OptimizerContext::OptimizerContextOptions optimizerOptions;

  DawnCompiler compiler;
  OptimizerContext optimizer(compiler.getDiagnostics(), optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Cartesian));

  // run single pass (PassLocalVarType)
  PassLocalVarType passLocalVarType(optimizer);
  passLocalVarType.run(stencil);

  int varAID = stencil->getMetaData().getAccessIDFromName("varA");
  int varBID = stencil->getMetaData().getAccessIDFromName("varB");
  ASSERT_TRUE(stencil->getMetaData().getLocalVariableDataFromAccessID(varAID).isTypeSet());
  // Need to check that varA has been flagged as IJ
  ASSERT_TRUE(stencil->getMetaData().getLocalVariableDataFromAccessID(varAID).getType() ==
              iir::LocalVariableType::OnIJ);
  // Need to check that varB has been flagged as IJ
  ASSERT_TRUE(stencil->getMetaData().getLocalVariableDataFromAccessID(varBID).getType() ==
              iir::LocalVariableType::OnIJ);
}

} // namespace