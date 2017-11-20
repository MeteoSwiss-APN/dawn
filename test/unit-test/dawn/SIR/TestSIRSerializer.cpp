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

#include "dawn/SIR/SIR.h"
#include "dawn/SIR/SIRSerializer.h"
#include <gtest/gtest.h>

using namespace dawn;

namespace {

#define SIR_EXCPECT_EQ(sir1, sir2)                                                                 \
  do {                                                                                             \
    auto comp = sir1->comparison(*sir2);                                                           \
    EXPECT_TRUE(bool(comp)) << comp.why();                                                         \
  } while(0);

class SIRSerializerTest : public ::testing::Test {
protected:
  virtual void SetUp() override { sirRef = std::make_shared<SIR>(); }
  virtual void TearDown() override { sirRef.reset(); }

  std::shared_ptr<SIR> serializeAndDeserializeRef() {
    return SIRSerializer::deserializeFromJsonString(SIRSerializer::serializeToJsonString(sirRef.get()));
  }

  std::shared_ptr<SIR> sirRef;
};

class SIRSerializerStencilTest : public SIRSerializerTest {
  virtual void SetUp() override {
    SIRSerializerTest::SetUp();

    sirRef->Stencils.emplace_back(std::make_shared<sir::Stencil>());
  }
};

TEST_F(SIRSerializerStencilTest, Name) {
  sirRef->Stencils[0]->Name = "foo";
  SIR_EXCPECT_EQ(sirRef, serializeAndDeserializeRef());
}

TEST_F(SIRSerializerStencilTest, SourceLocation) {
  sirRef->Stencils[0]->Loc = SourceLocation(5, 5);
  EXPECT_EQ(sirRef->Stencils[0]->Loc, serializeAndDeserializeRef()->Stencils[0]->Loc);
}

TEST_F(SIRSerializerStencilTest, Fields) {
  sirRef->Stencils[0]->Fields.emplace_back(std::make_shared<sir::Field>("foo"));
  sirRef->Stencils[0]->Fields.emplace_back(std::make_shared<sir::Field>("bar"));
  SIR_EXCPECT_EQ(sirRef, serializeAndDeserializeRef());
}

TEST_F(SIRSerializerStencilTest, AST) {
  sirRef->Stencils[0]->StencilDescAst =
      std::make_shared<AST>(std::make_shared<BlockStmt>(std::vector<std::shared_ptr<Stmt>>{
          std::make_shared<ExprStmt>(std::make_shared<FieldAccessExpr>("bar"))}));
  SIR_EXCPECT_EQ(sirRef, serializeAndDeserializeRef());
}

class SIRSerializerStencilFunctionTest : public SIRSerializerTest {
  virtual void SetUp() override {
    SIRSerializerTest::SetUp();

    sirRef->StencilFunctions.emplace_back(std::make_shared<sir::StencilFunction>());
  }
};

TEST_F(SIRSerializerStencilFunctionTest, Name) {
  sirRef->StencilFunctions[0]->Name = "foo";
  SIR_EXCPECT_EQ(sirRef, serializeAndDeserializeRef());
}

TEST_F(SIRSerializerStencilFunctionTest, SourceLocation) {
  sirRef->StencilFunctions[0]->Loc = SourceLocation(5, 5);
  EXPECT_EQ(sirRef->StencilFunctions[0]->Loc,
            serializeAndDeserializeRef()->StencilFunctions[0]->Loc);
}

TEST_F(SIRSerializerStencilFunctionTest, Arguments) {
  sirRef->StencilFunctions[0]->Args.emplace_back(std::make_shared<sir::Field>("foo"));
  sirRef->StencilFunctions[0]->Args.emplace_back(std::make_shared<sir::Offset>("foo"));
  sirRef->StencilFunctions[0]->Args.emplace_back(std::make_shared<sir::Direction>("foo"));
  SIR_EXCPECT_EQ(sirRef, serializeAndDeserializeRef());
}

TEST_F(SIRSerializerStencilFunctionTest, ASTs) {
  sirRef->StencilFunctions[0]->Asts.emplace_back(
      std::make_shared<AST>(std::make_shared<BlockStmt>(std::vector<std::shared_ptr<Stmt>>{
          std::make_shared<ExprStmt>(std::make_shared<FieldAccessExpr>("bar"))})));
  SIR_EXCPECT_EQ(sirRef, serializeAndDeserializeRef());
}

TEST_F(SIRSerializerStencilFunctionTest, Intervals) {
  sirRef->StencilFunctions[0]->Intervals.emplace_back(
      std::make_shared<sir::Interval>(sir::Interval::Start, sir::Interval::End));

  sirRef->StencilFunctions[0]->Intervals.emplace_back(
      std::make_shared<sir::Interval>(0, sir::Interval::End, -1, 2));

  sirRef->StencilFunctions[0]->Intervals.emplace_back(std::make_shared<sir::Interval>(5, 9, 1, -1));

  SIR_EXCPECT_EQ(sirRef, serializeAndDeserializeRef());
}

class SIRSerializerGlobalVariableTest : public SIRSerializerTest {
  virtual void SetUp() override {
    SIRSerializerTest::SetUp();

    sirRef->GlobalVariableMap = std::make_shared<sir::GlobalVariableMap>();
  }
};

TEST_F(SIRSerializerGlobalVariableTest, Value) {
  sirRef->GlobalVariableMap->emplace("int", std::make_shared<sir::Value>(5));
  sirRef->GlobalVariableMap->emplace("double", std::make_shared<sir::Value>(5.5));
  sirRef->GlobalVariableMap->emplace("string", std::make_shared<sir::Value>(std::string{"str"}));
  sirRef->GlobalVariableMap->emplace("bool", std::make_shared<sir::Value>(true));

  SIR_EXCPECT_EQ(sirRef, serializeAndDeserializeRef());
}

} // anonymous namespace
