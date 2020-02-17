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
#include "dawn/Optimizer/PassRemoveScalars.h"
#include "dawn/Unittest/IIRBuilder.h"

#include <gtest/gtest.h>

using namespace dawn;

namespace {

std::shared_ptr<iir::Expr> getRhsOfAssignment(const std::shared_ptr<iir::Stmt> stmt) {
  if(stmt->getKind() == iir::Stmt::Kind::VarDeclStmt) {
    const auto& varDeclStmt = std::dynamic_pointer_cast<iir::VarDeclStmt>(stmt);
    return varDeclStmt->getInitList()[0];
  } else if(stmt->getKind() == iir::Stmt::Kind::ExprStmt) {
    const auto& exprStmt = std::dynamic_pointer_cast<iir::ExprStmt>(stmt);
    if(exprStmt->getExpr()->getKind() == iir::Expr::Kind::AssignmentExpr) {
      const auto& assignmentExpr =
          std::dynamic_pointer_cast<iir::AssignmentExpr>(exprStmt->getExpr());
      return assignmentExpr->getRight();
    }
  }

  return nullptr;
}

iir::DoMethod& getDoMethod(std::shared_ptr<iir::StencilInstantiation>& si) {
  auto& iir = si->getIIR();
  auto& stencil = iir->getChild(0);
  auto& ms = stencil->getChild(0);
  auto& stage = ms->getChild(0);
  return *stage->getChild(0);
}

std::shared_ptr<iir::Stmt> getNthStmt(std::shared_ptr<iir::StencilInstantiation>& si, int n) {
  return getDoMethod(si).getAST().getStatements()[n];
}

bool isVarInDoMethodsAccesses(int varAccessID, const iir::DoMethod& doMethod) {
  for(const auto& stmt : doMethod.getAST().getStatements()) {
    const auto& access = stmt->getData<iir::IIRStmtData>().CallerAccesses;

    for(auto& accessPair : access->getWriteAccesses()) {
      if(varAccessID == accessPair.first) { // first = AccessID
        return true;
      }
    }
    for(const auto& accessPair : access->getReadAccesses()) {
      if(varAccessID == accessPair.first) { // first = AccessID
        return true;
      }
    }
  }
}

TEST(TestRemoveScalars, test_unstructured_scalar_01) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_c = b.field("f_c", ast::LocationType::Cells);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(3.0)});

  /// field(cells) f_c;
  /// double varA = 3.0;
  /// f_c = varA;

  auto stencil = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                             b.declareVar(varA), b.stmt(b.assignExpr(b.at(f_c), b.at(varA))))))));

  // Setup variable's metadata before running pass
  auto& metadata = stencil->getMetaData();
  int varAID = metadata.getAccessIDFromName("varA");
  metadata.getLocalVariableDataFromAccessID(varAID).setType(iir::LocalVariableType::Scalar);

  OptimizerContext::OptimizerContextOptions optimizerOptions;

  DawnCompiler compiler;
  OptimizerContext optimizer(compiler.getDiagnostics(), optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  PassRemoveScalars passRemoveScalars(optimizer);
  passRemoveScalars.run(stencil);

  // Check that there is 1 statement left
  ASSERT_EQ(getDoMethod(stencil).getAST().getStatements().size(), 1);

  auto firstStatement = getNthStmt(stencil, 0);
  // Check that first statement is: f_c = 3.0;
  ASSERT_EQ(firstStatement->getKind(), iir::Stmt::Kind::ExprStmt);
  ASSERT_TRUE(getRhsOfAssignment(firstStatement));
  ASSERT_EQ(getRhsOfAssignment(firstStatement)->getKind(), iir::Expr::Kind::LiteralAccessExpr);
  ASSERT_EQ(std::stof(std::dynamic_pointer_cast<iir::LiteralAccessExpr>(
                          getRhsOfAssignment(firstStatement))
                          ->getValue()),
            3.0);
  // Check that variable's metadata is gone
  ASSERT_EQ(metadata.getAccessIDToLocalVariableDataMap().count(varAID), 0);
  // Check that statements' accesses do not contain the variable
  ASSERT_FALSE(isVarInDoMethodsAccesses(varAID, getDoMethod(stencil)));
}

TEST(TestRemoveScalars, test_unstructured_scalar_02) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto f_c = b.field("f_c", ast::LocationType::Cells);
  auto f_c_out = b.field("f_c_out", ast::LocationType::Cells);
  auto varA = b.localvar("varA", dawn::BuiltinTypeID::Double, {b.lit(3.0)});
  auto varB = b.localvar("varB", dawn::BuiltinTypeID::Double, {b.lit(5.0)});

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

  // Setup variables' metadata before running pass
  auto& metadata = stencil->getMetaData();
  int varAID = metadata.getAccessIDFromName("varA");
  int varBID = metadata.getAccessIDFromName("varB");
  metadata.getLocalVariableDataFromAccessID(varAID).setType(iir::LocalVariableType::Scalar);
  metadata.getLocalVariableDataFromAccessID(varBID).setType(iir::LocalVariableType::Scalar);

  OptimizerContext::OptimizerContextOptions optimizerOptions;

  DawnCompiler compiler;
  OptimizerContext optimizer(compiler.getDiagnostics(), optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Unstructured));

  PassRemoveScalars passRemoveScalars(optimizer);
  passRemoveScalars.run(stencil);

  // Check that there are 2 statements left
  ASSERT_EQ(getDoMethod(stencil).getAST().getStatements().size(), 2);

  auto firstStatement = getNthStmt(stencil, 0);
  // Check that first statement is: f_c = 5.0 + (3.0 + 1.0);
  ASSERT_EQ(firstStatement->getKind(), iir::Stmt::Kind::ExprStmt);
  auto rhs = getRhsOfAssignment(firstStatement);
  ASSERT_TRUE(rhs);
  ASSERT_EQ(rhs->getKind(), iir::Expr::Kind::BinaryOperator);
  auto binOpExt = std::dynamic_pointer_cast<iir::BinaryOperator>(rhs);
  ASSERT_EQ(binOpExt->getOp(), Op::plus);
  ASSERT_EQ(binOpExt->getLeft()->getKind(), iir::Expr::Kind::LiteralAccessExpr);
  ASSERT_EQ(
      std::stof(std::dynamic_pointer_cast<iir::LiteralAccessExpr>(binOpExt->getLeft())->getValue()),
      5.0);
  ASSERT_EQ(binOpExt->getRight()->getKind(), iir::Expr::Kind::BinaryOperator);
  auto binOpInt = std::dynamic_pointer_cast<iir::BinaryOperator>(binOpExt->getRight());
  ASSERT_EQ(binOpInt->getOp(), Op::plus);
  ASSERT_EQ(
      std::stof(std::dynamic_pointer_cast<iir::LiteralAccessExpr>(binOpInt->getLeft())->getValue()),
      3.0);
  ASSERT_EQ(
      std::stof(
          std::dynamic_pointer_cast<iir::LiteralAccessExpr>(binOpInt->getRight())->getValue()),
      1.0);
  // Check that variables' metadata is gone
  ASSERT_EQ(metadata.getAccessIDToLocalVariableDataMap().count(varAID), 0);
  ASSERT_EQ(metadata.getAccessIDToLocalVariableDataMap().count(varBID), 0);
  // Check that statements' accesses do not contain the variables
  ASSERT_FALSE(isVarInDoMethodsAccesses(varAID, getDoMethod(stencil)));
  ASSERT_FALSE(isVarInDoMethodsAccesses(varBID, getDoMethod(stencil)));
}

// TODO TEST globals

// TODO TEST
// double varA = 2.0;
// if(f_c > 0.0) {
//   f_c_out = varA;
// }
//

// TODO TEST throw compound assignments
// TODO TEST throw i++;

// TODO TEST throw not supported
// double varA = 1.0;
// if(cond_adimensional) {
//    varA = 4.0;
// }
// f_c = varA;

} // namespace
