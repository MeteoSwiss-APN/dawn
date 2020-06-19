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
#include <gtest/gtest.h>
#include <string>

using namespace dawn;

namespace {

#define SIR_EXPECT_IMPL(sir1, sir2, VALUE)                                                         \
  do {                                                                                             \
    auto comp = sir1->comparison(*sir2);                                                           \
    EXPECT_##VALUE(bool(comp)) << comp.why();                                                      \
  } while(0);

#define SIR_EXCPECT_EQ(sir1, sir2) SIR_EXPECT_IMPL((sir1), (sir2), TRUE)
#define SIR_EXCPECT_NE(sir1, sir2) SIR_EXPECT_IMPL((sir1), (sir2), FALSE)

class SIRTest : public ::testing::Test {
protected:
  virtual void SetUp() override {
    sir1 = std::make_unique<SIR>(ast::GridType::Cartesian);
    sir2 = std::make_unique<SIR>(ast::GridType::Cartesian);
  }

  virtual void TearDown() override {
    sir1.release();
    sir2.release();
  }

  std::unique_ptr<SIR> sir1 = std::make_unique<SIR>(ast::GridType::Cartesian);
  std::unique_ptr<SIR> sir2 = std::make_unique<SIR>(ast::GridType::Cartesian);
};

class SIRStencilTest : public SIRTest {
  virtual void SetUp() override {
    SIRTest::SetUp();
    sir1->Stencils.emplace_back(std::make_shared<sir::Stencil>());
    sir2->Stencils.emplace_back(std::make_shared<sir::Stencil>());
  }
};

TEST_F(SIRStencilTest, Name) {
  sir1->Stencils[0]->Name = "foo";
  sir2->Stencils[0]->Name = "foo";
  SIR_EXCPECT_EQ(sir1, sir2);

  sir1->Stencils[0]->Name = "bar";
  SIR_EXCPECT_NE(sir1, sir2);
}

TEST_F(SIRStencilTest, Location) {
  sir1->Stencils[0]->Loc = SourceLocation(5, 5);
  sir2->Stencils[0]->Loc = SourceLocation(3, 3);

  // Location should be ignored
  SIR_EXCPECT_EQ(sir1, sir2);
}

TEST_F(SIRStencilTest, Fields) {
  auto makeFieldDimensions = []() -> sir::FieldDimensions {
    return sir::FieldDimensions(sir::HorizontalFieldDimension(ast::cartesian, {true, true}), true);
  };

  sir1->Stencils[0]->Fields.emplace_back(
      std::make_shared<sir::Field>("foo", makeFieldDimensions()));
  sir2->Stencils[0]->Fields.emplace_back(
      std::make_shared<sir::Field>("foo", makeFieldDimensions()));
  SIR_EXCPECT_EQ(sir1, sir2);

  // Fields are not equal
  sir1->Stencils[0]->Fields.emplace_back(
      std::make_shared<sir::Field>("bar", makeFieldDimensions()));
  SIR_EXCPECT_NE(sir1, sir2);

  // Fields are equal again but bar in sir2 is a temporary
  sir2->Stencils[0]->Fields.emplace_back(
      std::make_shared<sir::Field>("bar", makeFieldDimensions()));
  sir2->Stencils[0]->Fields.back()->IsTemporary = true;
  SIR_EXCPECT_NE(sir1, sir2);
}

TEST_F(SIRStencilTest, AST) {
  sir1->Stencils[0]->StencilDescAst =
      std::make_shared<sir::AST>(sir::makeBlockStmt(std::vector<std::shared_ptr<sir::Stmt>>{
          sir::makeExprStmt(std::make_shared<sir::FieldAccessExpr>("bar"))}));
  sir2->Stencils[0]->StencilDescAst =
      std::make_shared<sir::AST>(sir::makeBlockStmt(std::vector<std::shared_ptr<sir::Stmt>>{
          sir::makeExprStmt(std::make_shared<sir::FieldAccessExpr>("bar"))}));
  SIR_EXCPECT_EQ(sir1, sir2);

  sir1->Stencils[0]->StencilDescAst =
      std::make_shared<sir::AST>(sir::makeBlockStmt(std::vector<std::shared_ptr<sir::Stmt>>{
          sir::makeExprStmt(std::make_shared<sir::FieldAccessExpr>("bar"))}));
  sir2->Stencils[0]->StencilDescAst =
      std::make_shared<sir::AST>(sir::makeBlockStmt(std::vector<std::shared_ptr<sir::Stmt>>{
          sir::makeExprStmt(std::make_shared<sir::FieldAccessExpr>("foo"))}));
  SIR_EXCPECT_NE(sir1, sir2);
}

class SIRStencilFunctionTest : public SIRTest {
  virtual void SetUp() override {
    SIRTest::SetUp();
    sir1->StencilFunctions.emplace_back(std::make_shared<sir::StencilFunction>());
    sir2->StencilFunctions.emplace_back(std::make_shared<sir::StencilFunction>());
  }
};

TEST_F(SIRStencilFunctionTest, Name) {
  sir1->StencilFunctions[0]->Name = "foo";
  sir2->StencilFunctions[0]->Name = "foo";
  SIR_EXCPECT_EQ(sir1, sir2);

  sir1->StencilFunctions[0]->Name = "bar";
  SIR_EXCPECT_NE(sir1, sir2);
}

TEST_F(SIRStencilFunctionTest, Location) {
  sir1->StencilFunctions[0]->Loc = SourceLocation(5, 5);
  sir2->StencilFunctions[0]->Loc = SourceLocation(3, 3);

  // Location should be ignored
  SIR_EXCPECT_EQ(sir1, sir2);
}

TEST_F(SIRStencilFunctionTest, Arguments) {
  auto makeFieldDimensions = []() -> sir::FieldDimensions {
    return sir::FieldDimensions(sir::HorizontalFieldDimension(ast::cartesian, {true, true}), true);
  };

  sir1->StencilFunctions[0]->Args.emplace_back(
      std::make_shared<sir::Field>("foo", makeFieldDimensions()));
  sir2->StencilFunctions[0]->Args.emplace_back(
      std::make_shared<sir::Field>("foo", makeFieldDimensions()));
  SIR_EXCPECT_EQ(sir1, sir2);

  sir1->StencilFunctions[0]->Args.emplace_back(std::make_shared<sir::Offset>("foo"));
  SIR_EXCPECT_NE(sir1, sir2);

  sir2->StencilFunctions[0]->Args.emplace_back(std::make_shared<sir::Direction>("foo"));
  SIR_EXCPECT_NE(sir1, sir2);
}

TEST_F(SIRStencilFunctionTest, Interval) {
  sir1->StencilFunctions[0]->Intervals.emplace_back(
      std::make_shared<sir::Interval>(0, sir::Interval::End, 0, 0));
  sir2->StencilFunctions[0]->Intervals.emplace_back(
      std::make_shared<sir::Interval>(0, sir::Interval::End, 0, 0));
  SIR_EXCPECT_EQ(sir1, sir2);

  sir1->StencilFunctions[0]->Intervals.emplace_back(
      std::make_shared<sir::Interval>(0, sir::Interval::End, 0, 0));
  SIR_EXCPECT_NE(sir1, sir2);

  sir2->StencilFunctions[0]->Intervals.emplace_back(
      std::make_shared<sir::Interval>(0, sir::Interval::End, 1, 1));
  SIR_EXCPECT_NE(sir1, sir2);
}

TEST_F(SIRStencilFunctionTest, AST) {
  sir1->StencilFunctions[0]->Asts.emplace_back(
      std::make_shared<sir::AST>(sir::makeBlockStmt(std::vector<std::shared_ptr<sir::Stmt>>{
          sir::makeExprStmt(std::make_shared<sir::FieldAccessExpr>("bar"))})));
  sir2->StencilFunctions[0]->Asts.emplace_back(
      std::make_shared<sir::AST>(sir::makeBlockStmt(std::vector<std::shared_ptr<sir::Stmt>>{
          sir::makeExprStmt(std::make_shared<sir::FieldAccessExpr>("bar"))})));
  SIR_EXCPECT_EQ(sir1, sir2);

  sir1->StencilFunctions[0]->Asts.emplace_back(
      std::make_shared<sir::AST>(sir::makeBlockStmt(std::vector<std::shared_ptr<sir::Stmt>>{
          sir::makeExprStmt(std::make_shared<sir::FieldAccessExpr>("bar"))})));
  sir2->StencilFunctions[0]->Asts.emplace_back(
      std::make_shared<sir::AST>(sir::makeBlockStmt(std::vector<std::shared_ptr<sir::Stmt>>{
          sir::makeExprStmt(std::make_shared<sir::FieldAccessExpr>("foo"))})));
  SIR_EXCPECT_NE(sir1, sir2);
}

class SIRGlobalVariableTest : public SIRTest {
  virtual void SetUp() override {
    SIRTest::SetUp();
    sir1->GlobalVariableMap = std::make_shared<sir::GlobalVariableMap>();
    sir2->GlobalVariableMap = std::make_shared<sir::GlobalVariableMap>();
  }
};

TEST_F(SIRGlobalVariableTest, Comparison) {
  sir1->GlobalVariableMap->insert(std::pair("foo", sir::Global(5)));
  sir2->GlobalVariableMap->insert(std::pair("foo", sir::Global(5)));
  SIR_EXCPECT_EQ(sir1, sir2);

  // key is identical (not inserted)
  sir1->GlobalVariableMap->insert(std::pair("foo", sir::Global(5)));
  SIR_EXCPECT_EQ(sir1, sir2);

  // Different number of members
  sir1->GlobalVariableMap->insert(std::pair("bar", sir::Global(5.0)));
  SIR_EXCPECT_NE(sir1, sir2);

  // Different values
  sir2->GlobalVariableMap->insert(std::pair("bar", sir::Global(4.0)));
  SIR_EXCPECT_NE(sir1, sir2);

  // Different types
  sir2->GlobalVariableMap->erase("bar");
  sir2->GlobalVariableMap->insert(std::pair("bar", sir::Global(bool(true))));
  SIR_EXCPECT_NE(sir1, sir2);

  // // Same type/value
  sir2->GlobalVariableMap->erase("bar");
  sir2->GlobalVariableMap->insert(std::pair("bar", sir::Global(double(5.0))));
  SIR_EXCPECT_EQ(sir1, sir2);
}

TEST(SIRIntervalTest, Comparison) {
  sir::Interval i1(0, sir::Interval::End, 0, 0);
  sir::Interval i2(0, sir::Interval::End, 1, 0);
  sir::Interval i3(0, sir::Interval::End, 1, 1);
  sir::Interval i4(0, sir::Interval::End - 1, 1, 1);
  sir::Interval i5(1, sir::Interval::End - 1, 1, 1);

  EXPECT_EQ(i1, i1);

  EXPECT_EQ(i2, i2);
  EXPECT_NE(i1, i2);

  EXPECT_EQ(i3, i3);
  EXPECT_NE(i1, i3);

  EXPECT_EQ(i4, i4);
  EXPECT_NE(i1, i4);

  EXPECT_EQ(i5, i5);
  EXPECT_NE(i1, i5);
}

TEST(SIRValueTest, Construction) {
  // sir::Value empty;
  // EXPECT_TRUE(empty.empty());
  // EXPECT_EQ(empty.getType(), sir::Value::Kind::None);

  // Test Boolean
  sir::Value valueBoolean(bool(true));
  // valueBoolean.setValue(bool(true));
  EXPECT_EQ(valueBoolean.getType(), sir::Value::Kind::Boolean);
  EXPECT_EQ(valueBoolean.getValue<bool>(), true);

  // Test Integer
  sir::Value valueInteger(int(5));
  // valueInteger.setValue();
  EXPECT_EQ(valueInteger.getType(), sir::Value::Kind::Integer);
  EXPECT_EQ(valueInteger.getValue<int>(), 5);

  // Test Double
  sir::Value valueDouble(double(5.0));
  // valueDouble.setValue();
  EXPECT_EQ(valueDouble.getType(), sir::Value::Kind::Double);
  EXPECT_EQ(valueDouble.getValue<double>(), 5.0);

  // Test String
  sir::Value valueString(std::string("string"));
  // valueString.setValue(std::string("string"));
  EXPECT_EQ(valueString.getType(), sir::Value::Kind::String);
  EXPECT_EQ(valueString.getValue<std::string>(), "string");
}

} // anonymous namespace
