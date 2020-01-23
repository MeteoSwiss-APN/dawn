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

#include "dawn/SIR/ASTExpr.h"
#include "dawn/SIR/ASTStmt.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "dawn/Support/Type.h"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace dawn;

namespace {

#define SIR_EXCPECT_EQ(sir1, sir2)                                                                 \
  do {                                                                                             \
    auto comp = sir1->comparison(*sir2);                                                           \
    EXPECT_TRUE(bool(comp)) << comp.why();                                                         \
  } while(0);

class SIRSerializerTest : public ::testing::TestWithParam<SIRSerializer::Format> {
protected:
  virtual void SetUp() override { sirRef = std::make_shared<SIR>(ast::GridType::Cartesian); }
  virtual void TearDown() override { sirRef.reset(); }

  std::shared_ptr<SIR> serializeAndDeserializeRef() {
    return SIRSerializer::deserializeFromString(
        SIRSerializer::serializeToString(sirRef.get(), this->GetParam()), this->GetParam());
  }

  std::shared_ptr<SIR> sirRef;
};

class StencilTest : public SIRSerializerTest {
  virtual void SetUp() override {
    SIRSerializerTest::SetUp();

    sirRef->Stencils.emplace_back(std::make_shared<sir::Stencil>());
  }
};

TEST_P(StencilTest, Name) {
  sirRef->Stencils[0]->Name = "foo";
  SIR_EXCPECT_EQ(sirRef, serializeAndDeserializeRef());
}

TEST_P(StencilTest, SourceLocation) {
  sirRef->Stencils[0]->Loc = SourceLocation(5, 5);
  EXPECT_EQ(sirRef->Stencils[0]->Loc, serializeAndDeserializeRef()->Stencils[0]->Loc);
}

TEST_P(StencilTest, Fields) {
  auto makeFieldDimensions = []() -> sir::FieldDimensions {
    return sir::FieldDimensions(sir::HorizontalFieldDimension(ast::cartesian, {true, true}), true);
  };

  sirRef->Stencils[0]->Fields.emplace_back(
      std::make_shared<sir::Field>("foo", makeFieldDimensions()));
  sirRef->Stencils[0]->Fields.emplace_back(
      std::make_shared<sir::Field>("bar", makeFieldDimensions()));
  SIR_EXCPECT_EQ(sirRef, serializeAndDeserializeRef());
}

TEST_P(StencilTest, FieldsWithAttributes) {
  sirRef->Stencils[0]->Fields.emplace_back(std::make_shared<sir::Field>(
      "foo", sir::FieldDimensions(sir::HorizontalFieldDimension(dawn::ast::cartesian, {true, true}),
                                  true)));
  sirRef->Stencils[0]->Fields[0]->IsTemporary = true;
  sirRef->Stencils[0]->Fields[0]->Dimensions = sir::FieldDimensions(
      sir::HorizontalFieldDimension(dawn::ast::cartesian, {true, true}), false);
  SIR_EXCPECT_EQ(sirRef, serializeAndDeserializeRef());
}

TEST_P(StencilTest, AST) {
  sirRef->Stencils[0]->StencilDescAst =
      std::make_shared<sir::AST>(sir::makeBlockStmt(std::vector<std::shared_ptr<sir::Stmt>>{
          sir::makeExprStmt(std::make_shared<sir::FieldAccessExpr>("bar"))}));
  SIR_EXCPECT_EQ(sirRef, serializeAndDeserializeRef());
}

TEST_P(StencilTest, AST_Reduction) {
  const auto& reductionExpr = std::make_shared<sir::ReductionOverNeighborExpr>(
      "*", std::make_shared<sir::FieldAccessExpr>("rhs"),
      std::make_shared<sir::LiteralAccessExpr>("0.", BuiltinTypeID::Double),
      ast::LocationType::Cells, ast::LocationType::Cells);

  sirRef->Stencils[0]->StencilDescAst = std::make_shared<sir::AST>(sir::makeBlockStmt(
      std::vector<std::shared_ptr<sir::Stmt>>{sir::makeExprStmt(reductionExpr)}));

  SIR_EXCPECT_EQ(sirRef, serializeAndDeserializeRef());
}

TEST_P(StencilTest, AST_ReductionWeighted) {
  std::vector<sir::Value> weights{sir::Value(1.), sir::Value(2.), sir::Value(3.)};

  const auto& reductionExpr = std::make_shared<sir::ReductionOverNeighborExpr>(
      "*", std::make_shared<sir::FieldAccessExpr>("rhs"),
      std::make_shared<sir::LiteralAccessExpr>("0.", BuiltinTypeID::Double), weights,
      ast::LocationType::Cells, ast::LocationType::Cells);

  sirRef->Stencils[0]->StencilDescAst = std::make_shared<sir::AST>(sir::makeBlockStmt(
      std::vector<std::shared_ptr<sir::Stmt>>{sir::makeExprStmt(reductionExpr)}));

  SIR_EXCPECT_EQ(sirRef, serializeAndDeserializeRef());
}

INSTANTIATE_TEST_CASE_P(SIRSerializeTest, StencilTest,
                        ::testing::Values(SIRSerializer::Format::Json,
                                          SIRSerializer::Format::Byte));

class StencilFunctionTest : public SIRSerializerTest {
  virtual void SetUp() override {
    SIRSerializerTest::SetUp();

    sirRef->StencilFunctions.emplace_back(std::make_shared<sir::StencilFunction>());
  }
};

TEST_P(StencilFunctionTest, Name) {
  sirRef->StencilFunctions[0]->Name = "foo";
  SIR_EXCPECT_EQ(sirRef, serializeAndDeserializeRef());
}

TEST_P(StencilFunctionTest, SourceLocation) {
  sirRef->StencilFunctions[0]->Loc = SourceLocation(5, 5);
  EXPECT_EQ(sirRef->StencilFunctions[0]->Loc,
            serializeAndDeserializeRef()->StencilFunctions[0]->Loc);
}

TEST_P(StencilFunctionTest, Arguments) {
  auto makeFieldDimensions = []() -> sir::FieldDimensions {
    return sir::FieldDimensions(sir::HorizontalFieldDimension(ast::cartesian, {true, true}), true);
  };

  sirRef->StencilFunctions[0]->Args.emplace_back(
      std::make_shared<sir::Field>("foo", makeFieldDimensions()));
  sirRef->StencilFunctions[0]->Args.emplace_back(std::make_shared<sir::Offset>("foo"));
  sirRef->StencilFunctions[0]->Args.emplace_back(std::make_shared<sir::Direction>("foo"));
  SIR_EXCPECT_EQ(sirRef, serializeAndDeserializeRef());
}

TEST_P(StencilFunctionTest, ASTsFieldAccess) {
  sirRef->StencilFunctions[0]->Asts.emplace_back(
      std::make_shared<sir::AST>(sir::makeBlockStmt(std::vector<std::shared_ptr<sir::Stmt>>{
          sir::makeExprStmt(std::make_shared<sir::FieldAccessExpr>("bar"))})));
  SIR_EXCPECT_EQ(sirRef, serializeAndDeserializeRef());
}

TEST_P(StencilFunctionTest, ASTsLiteralAccess) {
  sirRef->StencilFunctions[0]->Asts.emplace_back(std::make_shared<sir::AST>(
      sir::makeBlockStmt(std::vector<std::shared_ptr<sir::Stmt>>{sir::makeExprStmt(
          std::make_shared<sir::LiteralAccessExpr>("0.", BuiltinTypeID::Double))})));
  SIR_EXCPECT_EQ(sirRef, serializeAndDeserializeRef());
}

TEST_P(StencilFunctionTest, Intervals) {
  sirRef->StencilFunctions[0]->Intervals.emplace_back(
      std::make_shared<sir::Interval>(sir::Interval::Start, sir::Interval::End));

  sirRef->StencilFunctions[0]->Intervals.emplace_back(
      std::make_shared<sir::Interval>(0, sir::Interval::End, -1, 2));

  sirRef->StencilFunctions[0]->Intervals.emplace_back(std::make_shared<sir::Interval>(5, 9, 1, -1));

  SIR_EXCPECT_EQ(sirRef, serializeAndDeserializeRef());
}

INSTANTIATE_TEST_CASE_P(SIRSerializeTest, StencilFunctionTest,
                        ::testing::Values(SIRSerializer::Format::Json,
                                          SIRSerializer::Format::Byte));

class GlobalVariableTest : public SIRSerializerTest {
  virtual void SetUp() override {
    SIRSerializerTest::SetUp();

    sirRef->GlobalVariableMap = std::make_shared<sir::GlobalVariableMap>();
  }
};

TEST_P(GlobalVariableTest, Value) {
  sirRef->GlobalVariableMap->insert(std::pair("int", sir::Global(5)));
  sirRef->GlobalVariableMap->insert(std::pair("double", sir::Global(5.5)));
  sirRef->GlobalVariableMap->insert(std::pair("string", sir::Global(std::string{"str"})));
  sirRef->GlobalVariableMap->insert(std::pair("bool", sir::Global(true)));

  SIR_EXCPECT_EQ(sirRef, serializeAndDeserializeRef());
}

INSTANTIATE_TEST_CASE_P(SIRSerializeTest, GlobalVariableTest,
                        ::testing::Values(SIRSerializer::Format::Json,
                                          SIRSerializer::Format::Byte));

} // anonymous namespace
