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

#include "dawn/AST/ASTStringifier.h"
#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/Compiler/Options.h"
#include "dawn/IIR/ASTFwd.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/PassSimplifyStatements.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Unittest/ASTConstructionAliases.h"
#include "dawn/Unittest/CompilerUtil.h"
#include "dawn/Unittest/UnittestUtils.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"

#include <fstream>
#include <gtest/gtest.h>
#include <memory>

using namespace dawn;
using namespace astgen;

namespace {
class TestPassSimplifyStatements : public ::testing::Test {
protected:
  std::unique_ptr<OptimizerContext> context_;
  std::shared_ptr<iir::StencilInstantiation> instantiation_;

  void runPass(const std::string& filename) {
    dawn::UIDGenerator::getInstance()->reset();
    instantiation_ = IIRSerializer::deserialize(filename);

    PassSimplifyStatements pass(*context_);
    ASSERT_TRUE(pass.run(instantiation_));
  }
};

TEST_F(TestPassSimplifyStatements, CompoundStatement) {
  // b += a;
  // d -= c;
  runPass("input/test_simplify_statements_compound_statement.iir");
  auto const& firstStmt = getNthStmt(getFirstDoMethod(instantiation_), 0);
  ASSERT_TRUE(firstStmt->equals(expr(assign(field("b"), binop(field("b"), "+", field("a")))).get(),
                                /*compareData = */ false));
  auto const& secondStmt = getNthStmt(getFirstDoMethod(instantiation_), 1);
  ASSERT_TRUE(secondStmt->equals(expr(assign(field("d"), binop(field("d"), "-", field("c")))).get(),
                                 /*compareData = */ false));
}

TEST_F(TestPassSimplifyStatements, IncrementDecrement) {
  // int b = d;
  // int c = d;
  // --b;
  // ++c;
  // a = c + b;
  runPass("input/test_simplify_statements_increment_decrement.iir");
  ASSERT_EQ(5, getFirstDoMethod(instantiation_).getAST().getStatements().size());
  auto const& firstStmt = getNthStmt(getFirstDoMethod(instantiation_), 2);
  ASSERT_TRUE(firstStmt->equals(expr(assign(var("b"), binop(var("b"), "-", lit(1)))).get(),
                                /*compareData = */ false));
  auto const& secondStmt = getNthStmt(getFirstDoMethod(instantiation_), 3);
  ASSERT_TRUE(secondStmt->equals(expr(assign(var("c"), binop(var("c"), "+", lit(1)))).get(),
                                 /*compareData = */ false));
}

TEST_F(TestPassSimplifyStatements, IncrementNested) {
  // int b = c;
  // a = (++b) + 1;
  runPass("input/test_simplify_statements_increment_nested.iir");
  ASSERT_EQ(3, getFirstDoMethod(instantiation_).getAST().getStatements().size());
  auto const& firstStmt = getNthStmt(getFirstDoMethod(instantiation_), 1);
  ASSERT_TRUE(firstStmt->equals(expr(assign(var("b"), binop(var("b"), "+", lit(1)))).get(),
                                /*compareData = */ false));
  auto const& secondStmt = getNthStmt(getFirstDoMethod(instantiation_), 2);
  ASSERT_TRUE(secondStmt->equals(expr(assign(field("a"), binop(var("b"), "+", lit(1)))).get(),
                                 /*compareData = */ false));
}

TEST_F(TestPassSimplifyStatements, MixMultipleNested) {
  // int b = d;
  // int c = d;
  // a += ++b + (1 + --c);
  // a *= ++c * --b;
  runPass("input/test_simplify_statements_mix_multiple_nested.iir");
  ASSERT_EQ(8, getFirstDoMethod(instantiation_).getAST().getStatements().size());
  auto const& firstStmt = getNthStmt(getFirstDoMethod(instantiation_), 2);
  ASSERT_TRUE(firstStmt->equals(expr(assign(var("b"), binop(var("b"), "+", lit(1)))).get(),
                                /*compareData = */ false));
  auto const& secondStmt = getNthStmt(getFirstDoMethod(instantiation_), 3);
  ASSERT_TRUE(secondStmt->equals(expr(assign(var("c"), binop(var("c"), "-", lit(1)))).get(),
                                 /*compareData = */ false));
  auto const& thirdStmt = getNthStmt(getFirstDoMethod(instantiation_), 4);
  ASSERT_TRUE(thirdStmt->equals(
      expr(assign(field("a"),
                  binop(field("a"), "+", binop(var("b"), "+", binop(lit(1), "+", var("c"))))))
          .get(),
      /*compareData = */ false));
  auto const& fourthStmt = getNthStmt(getFirstDoMethod(instantiation_), 5);
  ASSERT_TRUE(fourthStmt->equals(expr(assign(var("c"), binop(var("c"), "+", lit(1)))).get(),
                                 /*compareData = */ false));
  auto const& fifthStmt = getNthStmt(getFirstDoMethod(instantiation_), 6);
  ASSERT_TRUE(fifthStmt->equals(expr(assign(var("b"), binop(var("b"), "-", lit(1)))).get(),
                                /*compareData = */ false));
  auto const& sixthStmt = getNthStmt(getFirstDoMethod(instantiation_), 7);
  ASSERT_TRUE(sixthStmt->equals(
      expr(assign(field("a"), binop(field("a"), "*", binop(var("c"), "*", var("b"))))).get(),
      /*compareData = */ false));
}

} // namespace
