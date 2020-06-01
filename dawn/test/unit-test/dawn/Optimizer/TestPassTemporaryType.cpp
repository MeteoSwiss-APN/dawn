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
#include "dawn/Optimizer/PassStageSplitter.h"
#include "dawn/Optimizer/PassTemporaryType.h"
#include "dawn/Serialization/IIRSerializer.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

class TestPassTemporaryType : public ::testing::Test {
protected:
  std::unique_ptr<OptimizerContext> context_;
  OptimizerContext::OptimizerContextOptions options_;

  explicit TestPassTemporaryType() {
    context_ = std::make_unique<OptimizerContext>(options_,
                                                  std::make_shared<SIR>(ast::GridType::Cartesian));
    UIDGenerator::getInstance()->reset();
  }

  void runTest(const std::string& filename, const std::unordered_set<std::string>& demotedFields,
               const std::unordered_set<std::string>& promotedFields = {}) {
    // Deserialize IIR
    auto instantiation = IIRSerializer::deserialize(filename);

    // Run stage splitter pass
    PassStageSplitter stageSplitPass(*context_);
    EXPECT_TRUE(stageSplitPass.run(instantiation));

    // Expect pass to succeed...
    PassTemporaryType tempTypePass(*context_);
    EXPECT_TRUE(tempTypePass.run(instantiation));

    // Apply AST matcher to find all field access expressions
    dawn::iir::ASTMatcher matcher(instantiation.get());
    std::vector<std::shared_ptr<ast::Expr>>& accessExprs =
        matcher.match(ast::Expr::Kind::FieldAccessExpr);

    std::unordered_set<std::string> fieldNames;
    for(const auto& accessExpr : accessExprs) {
      const auto& fieldAccessExpr = std::dynamic_pointer_cast<ast::FieldAccessExpr>(accessExpr);
      std::string fieldName = fieldAccessExpr->getName();
      if(fieldName.find("__tmp_") == 0) {
        // Undo temporary renaming...
        fieldName = fieldName.substr(6);
        fieldName = fieldName.substr(0, fieldName.rfind('_'));
      }
      fieldNames.insert(fieldName);
    }

    // Assert that demoted fields are not accessed
    for(const auto& demotedField : demotedFields) {
      ASSERT_TRUE(fieldNames.find(demotedField) == fieldNames.end());
    }

    // Assert that promoted fields are accessed
    for(const auto& promotedField : promotedFields) {
      ASSERT_TRUE(fieldNames.find(promotedField) != fieldNames.end());
    }
  }
};

TEST_F(TestPassTemporaryType, DemoteTest1) {
  /*
    vertical_region(k_start, k_end) {
      tmp = 5.0;
      foo = tmp;
    } */
  runTest("input/DemoteTest01.iir", {"tmp"});
}

TEST_F(TestPassTemporaryType, PromoteTest1) {
  /*
    vertical_region(k_start, k_end) {
      double local_variable = 5.0;
      field_a = field_b;
      field_c = field_a(i + 1) + local_variable;
    } */
  runTest("input/PromoteTest01.iir", {}, {"local_variable"});
}

TEST_F(TestPassTemporaryType, PromoteTest2) {
  /*
    vertical_region(k_start, k_end) {
      double local_variable = 5.0;
      field_a = field_b;
      field_c = field_a(i + 1);
      field_a = field_b;
      field_c = field_a(i + 1) + local_variable;
    } */
  runTest("input/PromoteTest02.iir", {}, {"local_variable"});
}

TEST_F(TestPassTemporaryType, PromoteTest3) {
  /*
    vertical_region(k_start, k_end) {
      double local_variable = 5.0;
      field_b = field_a;
      field_c = field_b(k - 1) + local_variable;
      field_d = field_c(k + 1) + local_variable;
    } */
  runTest("input/PromoteTest03.iir", {}, {"local_variable"});
}

TEST_F(TestPassTemporaryType, PromoteTest4) {
  /*
    vertical_region(k_start, k_end) {
      double local_variable = 5.0;
      local_variable *= local_variable;
      field_a = field_b;
      field_c = field_a(i + 1) + local_variable * local_variable;
      field_a = field_b;
      field_c = field_a(i + 1) + local_variable * local_variable;
    } */
  runTest("input/PromoteTest04.iir", {}, {"local_variable"});
}

TEST_F(TestPassTemporaryType, PromoteTest5) {
  /*
    vertical_region(k_start, k_end) {
      double local_variable = 5.0;
      local_variable *= local_variable;
      field_b = field_a;
      field_c = field_b(k - 1) + local_variable * local_variable;
      field_a = field_c(k + 1) + local_variable * local_variable;

      field_b = field_a;
      field_c = field_b(k - 1) + local_variable * local_variable;
      field_a = field_c(k + 1) + local_variable * local_variable;
    } */
  runTest("input/PromoteTest05.iir", {"field_a_0"}, {"local_variable"});
}

} // anonymous namespace
