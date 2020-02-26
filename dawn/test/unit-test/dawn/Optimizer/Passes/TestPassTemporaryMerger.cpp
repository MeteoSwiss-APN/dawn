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
#include "dawn/Compiler/Options.h"
#include "dawn/IIR/ASTMatcher.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/PassTemporaryMerger.h"
#include "dawn/Serialization/IIRSerializer.h"
//#include "dawn/Unittest/CompilerUtil.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

class TestPassTemporaryMerger : public ::testing::Test {
protected:
  dawn::OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_;
  dawn::DiagnosticsEngine diag_;

  explicit TestPassTemporaryMerger() {
    options_.MergeTemporaries = true;
    std::shared_ptr<SIR> sir = std::make_shared<SIR>(ast::GridType::Cartesian);
    context_ = std::make_unique<OptimizerContext>(diag_, options_, sir);
    dawn::UIDGenerator::getInstance()->reset();
  }

  void runTest(const std::string& filename, const std::vector<std::string>& mergedFields) {
    std::string filepath = filename;
    if(!TestEnvironment::path_.empty()) {
      filepath = TestEnvironment::path_ + "/" + filepath;
    }

    auto instantiation = IIRSerializer::deserialize(filepath);

    // Run prerequisite groups
//    ASSERT_TRUE(CompilerUtil::runGroup(PassGroup::Parallel, context_, instantiation));
//    ASSERT_TRUE(CompilerUtil::runGroup(PassGroup::ReorderStages, context_, instantiation));
//    ASSERT_TRUE(CompilerUtil::runGroup(PassGroup::MergeStages, context_, instantiation));

    // Expect pass to succeed...
    PassTemporaryMerger tempMergerPass(*context_);
    EXPECT_TRUE(tempMergerPass.run(instantiation));
    //ASSERT_TRUE(CompilerUtil::runPass<dawn::PassTemporaryMerger>(context_, instantiation));

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

TEST_F(TestPassTemporaryMerger, MergeTest1) { runTest("input/MergeTest01.iir", {}); }

TEST_F(TestPassTemporaryMerger, MergeTest2) { runTest("input/MergeTest02.iir", {}); }

TEST_F(TestPassTemporaryMerger, MergeTest3) { runTest("input/MergeTest03.iir", {"tmp_b"}); }

TEST_F(TestPassTemporaryMerger, MergeTest4) { runTest("input/MergeTest04.iir", {"tmp_b"}); }

TEST_F(TestPassTemporaryMerger, MergeTest5) {
  runTest("input/MergeTest05.iir", {"tmp_2", "tmp_3", "tmp_4", "tmp_5"});
}

} // anonymous namespace
