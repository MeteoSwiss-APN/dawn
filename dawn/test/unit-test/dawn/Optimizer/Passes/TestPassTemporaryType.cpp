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

#include "dawn/AST/ASTMatcher.h"
#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/Compiler/Options.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/PassTemporaryType.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Unittest/CompilerUtil.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

class TestPassTemporaryType : public ::testing::Test {
protected:
  dawn::OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_;

  void runTest(const std::string& filename, const std::vector<std::string>& demotedFields) {
    dawn::UIDGenerator::getInstance()->reset();
    std::shared_ptr<iir::StencilInstantiation> instantiation =
        CompilerUtil::load(filename, options_, context_, TestEnvironment::path_);

    // Run prerequisite tests
    ASSERT_TRUE(CompilerUtil::runPasses(5, context_, instantiation));

    //CompilerUtil::write(context_, 0, 1, "DemoteTest01");

    // Expect pass to succeed...
    ASSERT_TRUE(CompilerUtil::runPass<dawn::PassTemporaryType>(context_, instantiation));

    //CompilerUtil::write(context_, 1, 1, "DemoteTest01");

    if(demotedFields.size() > 0) {
      // Apply AST matcher to find all field access expressions
      dawn::ASTMatcher matcher(instantiation.get());
      std::vector<std::shared_ptr<ast::Expr>>& accessExprs =
          matcher.match(ast::Expr::Kind::FieldAccessExpr);

      std::unordered_set<std::string> fieldNames;
      for(const auto& access : accessExprs) {
        const auto& field = std::dynamic_pointer_cast<ast::FieldAccessExpr>(access);
        fieldNames.insert(field->getName());
      }

      // Assert that merged fields are no longer accessed
      for(const auto& demotedField : demotedFields) {
        ASSERT_TRUE(fieldNames.find(demotedField) == fieldNames.end());
      }
    }
  }
};

TEST_F(TestPassTemporaryType, DemoteTest1) { runTest("DemoteTest01.sir", {"tmp"}); }

//TEST_F(TestPassTemporaryType, PromoteTest1) { runTest("PromoteTest01.sir", {}); }
//
//TEST_F(TestPassTemporaryType, PromoteTest2) { runTest("PromoteTest02.sir", {}); }
//
//TEST_F(TestPassTemporaryType, PromoteTest3) { runTest("PromoteTest03.sir", {}); }
//
//TEST_F(TestPassTemporaryType, PromoteTest4) { runTest("PromoteTest04.sir", {}); }
//
//TEST_F(TestPassTemporaryType, PromoteTest5) { runTest("PromoteTest05.sir", {}); }

} // anonymous namespace