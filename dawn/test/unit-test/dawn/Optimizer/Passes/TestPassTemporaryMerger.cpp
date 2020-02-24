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
#include "dawn/Optimizer/PassTemporaryMerger.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Unittest/CompilerUtil.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

class TestPassTemporaryMerger : public ::testing::Test {
protected:
  dawn::OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_;

  virtual void SetUp() { options_.MergeTemporaries = true; }

  void runTest(const std::string& filename, const std::vector<std::string>& mergedFields) {
    dawn::UIDGenerator::getInstance()->reset();
    std::shared_ptr<iir::StencilInstantiation> instantiation =
        CompilerUtil::load(filename, options_, context_, TestEnvironment::path_);

    // Run prerequisite groups
    ASSERT_TRUE(CompilerUtil::runGroup(PassGroup::Parallel, context_, instantiation));
    ASSERT_TRUE(CompilerUtil::runGroup(PassGroup::ReorderStages, context_, instantiation));
    ASSERT_TRUE(CompilerUtil::runGroup(PassGroup::MergeStages, context_, instantiation));

    // Expect pass to succeed...
    ASSERT_TRUE(CompilerUtil::runPass<dawn::PassTemporaryMerger>(context_, instantiation));

    if(mergedFields.size() > 0) {
      // Apply AST matcher to find all field access expressions
      dawn::iir::ASTMatcher matcher(instantiation.get());
      std::vector<std::shared_ptr<ast::Expr>>& accessExprs =
          matcher.match(ast::Expr::Kind::FieldAccessExpr);

      std::unordered_set<std::string> fieldNames;
      for(const auto& access : accessExprs) {
        const auto& fieldAccessExpr = std::dynamic_pointer_cast<ast::FieldAccessExpr>(access);
        fieldNames.insert(fieldAccessExpr->getName());
      }

      // Assert that merged fields are no longer accessed
      for(const auto& mergedField : mergedFields) {
        ASSERT_TRUE(fieldNames.find(mergedField) == fieldNames.end());
      }
    }
  }
};

TEST_F(TestPassTemporaryMerger, MergeTest1) { runTest("input/MergeTest01.sir", {}); }

TEST_F(TestPassTemporaryMerger, MergeTest2) { runTest("input/MergeTest02.sir", {}); }

TEST_F(TestPassTemporaryMerger, MergeTest3) { runTest("input/MergeTest03.sir", {"tmp_b"}); }

TEST_F(TestPassTemporaryMerger, MergeTest4) { runTest("input/MergeTest04.sir", {"tmp_b"}); }

TEST_F(TestPassTemporaryMerger, MergeTest5) {
  runTest("input/MergeTest05.sir", {"tmp_2", "tmp_3", "tmp_4", "tmp_5"});
}

} // anonymous namespace
