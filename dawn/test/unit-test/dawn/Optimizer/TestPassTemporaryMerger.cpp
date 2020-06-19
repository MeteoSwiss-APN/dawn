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

#include "dawn/IIR/ASTMatcher.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassSetStageGraph.h"
#include "dawn/Optimizer/PassStageSplitter.h"
#include "dawn/Optimizer/PassTemporaryMerger.h"
#include "dawn/Serialization/IIRSerializer.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

class TestPassTemporaryMerger : public ::testing::Test {
protected:
  OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_;

  explicit TestPassTemporaryMerger() {
    context_ = std::make_unique<OptimizerContext>(options_,
                                                  std::make_shared<SIR>(ast::GridType::Cartesian));
    UIDGenerator::getInstance()->reset();
  }

  void runTest(const std::string& filename,
               const std::unordered_set<std::string>& mergedFields = {}) {
    // Deserialize IIR
    auto instantiation = IIRSerializer::deserialize(filename);

    // Run stage splitter pass
    PassStageSplitter stageSplitPass(*context_);
    EXPECT_TRUE(stageSplitPass.run(instantiation));

    // Expect pass to succeed...
    PassTemporaryMerger tempMergerPass(*context_);
    EXPECT_TRUE(tempMergerPass.run(instantiation));

    if(mergedFields.size() > 0) {
      // Apply AST matcher to find all field access expressions
      dawn::iir::ASTMatcher matcher(instantiation.get());
      std::vector<std::shared_ptr<ast::Expr>>& accessExprs =
          matcher.match(ast::Expr::Kind::FieldAccessExpr);

      std::unordered_set<std::string> fieldNames;
      for(const auto& accessExpr : accessExprs) {
        const auto& fieldAccessExpr = std::dynamic_pointer_cast<ast::FieldAccessExpr>(accessExpr);
        fieldNames.insert(fieldAccessExpr->getName());
      }

      // Assert that merged fields are no longer accessed
      for(const auto& mergedField : mergedFields) {
        ASSERT_TRUE(fieldNames.find(mergedField) == fieldNames.end());
      }
    }
  }
};

TEST_F(TestPassTemporaryMerger, MergeTest1) {
  /*
   vertical_region(k_start, k_end) { field_a = field_b; }
   */
  runTest("input/MergeTest01.iir");
}

TEST_F(TestPassTemporaryMerger, MergeTest2) {
  /*
    vertical_region(k_start, k_end) {
      tmp_a = field_a;
      tmp_b = field_b;
      field_a = tmp_a(i + 1);
      field_b = tmp_b(i + 1);
    } */
  runTest("input/MergeTest02.iir");
}

TEST_F(TestPassTemporaryMerger, MergeTest3) {
  /*
    vertical_region(k_start, k_end) {
      tmp_a = field_a;
      field_a = tmp_a(i + 1);
      tmp_b = field_b;
      field_b = tmp_b(i + 1);
      } */
  runTest("input/MergeTest03.iir", {"tmp_b"});
}

TEST_F(TestPassTemporaryMerger, MergeTest4) {
  /*
    vertical_region(k_start, k_end) {
      tmp_a = field_a;
      field_a = tmp_a(i + 1);
    }
    vertical_region(k_start, k_end) {
      tmp_b = field_b;
      field_b = tmp_b(i + 1);
    } */
  runTest("input/MergeTest04.iir", {"tmp_b"});
}

TEST_F(TestPassTemporaryMerger, MergeTest5) {
  /*
    vertical_region(k_start, k_end) {
      tmp_1 = field_1;
      field_1 = tmp_1(i + 1);
      tmp_2 = field_2;
      field_2 = tmp_2(i + 1);
      tmp_3 = field_3;
      field_3 = tmp_3(i + 1);
      tmp_4 = field_4;
      field_4 = tmp_4(i + 1);
      tmp_5 = field_5;
      field_5 = tmp_5(i + 1);
    } */
  runTest("input/MergeTest05.iir", {"tmp_2", "tmp_3", "tmp_4", "tmp_5"});
}

} // anonymous namespace
