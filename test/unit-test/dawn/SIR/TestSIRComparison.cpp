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

#include "dawn/SIR/AST.h"
#include "dawn/SIR/ASTUtil.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/STLExtras.h"
#include <gtest/gtest.h>
#include <iostream>

using namespace dawn;
using namespace sir;

namespace {

TEST(SIRComparisonTest, DoubleReferenceFullSIR) {
  std::unique_ptr<SIR> test01SIR = make_unique<SIR>();
  std::unique_ptr<SIR> test02SIR = make_unique<SIR>();

  // Genereate a stencil
  test01SIR->Stencils.resize(1);
  test02SIR->Stencils.resize(1);
  test01SIR->Stencils[0] = std::make_shared<Stencil>();
  test02SIR->Stencils[0] = std::make_shared<Stencil>();
  test01SIR->Stencils[0]->Name = "TestStencil01";
  test02SIR->Stencils[0]->Name = "TestStencil01";

  auto astptr =
      std::make_shared<dawn::AST>(std::make_shared<BlockStmt>(std::vector<std::shared_ptr<Stmt>>{
          std::make_shared<ExprStmt>(std::make_shared<VarAccessExpr>("foo1"))}));
  test01SIR->Stencils[0]->StencilDescAst = astptr;
  test02SIR->Stencils[0]->StencilDescAst =
      std::make_shared<dawn::AST>(std::make_shared<BlockStmt>(std::vector<std::shared_ptr<Stmt>>{
          std::make_shared<ExprStmt>(std::make_shared<VarAccessExpr>("foo1"))}));

  test01SIR->Stencils[0]->Fields.resize(1);
  test02SIR->Stencils[0]->Fields.resize(1);
  test01SIR->Stencils[0]->Fields[0] = std::make_shared<sir::Field>("foo2");
  test02SIR->Stencils[0]->Fields[0] = std::make_shared<sir::Field>("foo2");

  test01SIR->Stencils[0]->Attributes.set(Attr::AttrKind::AK_MergeTemporaries);
  test02SIR->Stencils[0]->Attributes.set(Attr::AttrKind::AK_MergeTemporaries);

  // Generate a stencil Function
  test01SIR->StencilFunctions.resize(1);
  for(auto& a : test01SIR->StencilFunctions) {
    a = std::make_shared<StencilFunction>();
  }
  test02SIR->StencilFunctions.resize(1);
  for(auto& a : test02SIR->StencilFunctions) {
    a = std::make_shared<StencilFunction>();
  }
  test01SIR->StencilFunctions[0]->Name = "TestStencilFunction01";
  test02SIR->StencilFunctions[0]->Name = "TestStencilFunction01";

  test01SIR->StencilFunctions[0]->Args.resize(1);
  test02SIR->StencilFunctions[0]->Args.resize(1);

  test01SIR->StencilFunctions[0]->Args[0] = std::make_shared<Field>("foo");
  test02SIR->StencilFunctions[0]->Args[0] = std::make_shared<Field>("foo");

  test01SIR->StencilFunctions[0]->Intervals.resize(1);
  test02SIR->StencilFunctions[0]->Intervals.resize(1);
  test01SIR->StencilFunctions[0]->Intervals[0] = std::make_shared<Interval>(0, 1);
  ;
  test02SIR->StencilFunctions[0]->Intervals[0] = std::make_shared<Interval>(0, 1);
  ;

  test01SIR->StencilFunctions[0]->Asts.resize(1);
  test02SIR->StencilFunctions[0]->Asts.resize(1);
  test01SIR->StencilFunctions[0]->Asts[0] =
      std::make_shared<AST>(std::make_shared<BlockStmt>(std::vector<std::shared_ptr<Stmt>>{
          std::make_shared<ExprStmt>(std::make_shared<FieldAccessExpr>("bar"))}));
  test02SIR->StencilFunctions[0]->Asts[0] =
      std::make_shared<AST>(std::make_shared<BlockStmt>(std::vector<std::shared_ptr<Stmt>>{
          std::make_shared<ExprStmt>(std::make_shared<FieldAccessExpr>("bar"))}));

  test01SIR->StencilFunctions[0]->Attributes.set(Attr::AttrKind::AK_MergeTemporaries);
  test02SIR->StencilFunctions[0]->Attributes.set(Attr::AttrKind::AK_MergeTemporaries);

  // Generate a global Variable Map
  test02SIR->GlobalVariableMap = std::make_shared<GlobalVariableMap>();
  test01SIR->GlobalVariableMap = std::make_shared<GlobalVariableMap>();
  auto val01 = std::make_shared<Value>();
  val01.get()->setType(Value::TypeKind::Integer);
  val01.get()->setValue(10);
  auto val02 = std::make_shared<Value>();
  val02.get()->setType(Value::TypeKind::Integer);
  val02.get()->setValue(10);
  test01SIR->GlobalVariableMap->emplace("TestGlobal", val01);
  test02SIR->GlobalVariableMap->emplace("TestGlobal", val02);

  auto a = (*test01SIR).comparison(*test02SIR);
  EXPECT_TRUE(a.second) << a.first;
}

TEST(SIRComparisonTest, Errorcheck) {
  std::unique_ptr<SIR> test01SIR = make_unique<SIR>();
  std::unique_ptr<SIR> test02SIR = make_unique<SIR>();

  // Genereate a stencil
  test01SIR->Stencils.resize(1);
  test02SIR->Stencils.resize(1);
  test01SIR->Stencils[0] = std::make_shared<Stencil>();
  test02SIR->Stencils[0] = std::make_shared<Stencil>();
  test01SIR->Stencils[0]->Name = "TestStencil01";
  test02SIR->Stencils[0]->Name = "TestStencil01";

  // Generate a global Variable Map
  test02SIR->GlobalVariableMap = std::make_shared<GlobalVariableMap>();
  test01SIR->GlobalVariableMap = std::make_shared<GlobalVariableMap>();
  auto val01 = std::make_shared<Value>();
  val01.get()->setType(Value::TypeKind::Integer);
  val01.get()->setValue(10);
  auto val02 = std::make_shared<Value>();
  val02.get()->setType(Value::TypeKind::Integer);
  val02.get()->setValue(12);
  test01SIR->GlobalVariableMap->emplace("TestGlobal", val01);
  test02SIR->GlobalVariableMap->emplace("TestGlobal", val02);

  auto a = (*test01SIR).comparison(*test02SIR);
//  EXPECT_TRUE(a.second) << a.first;
  ASSERT_FALSE(a.second) << "SIRs match but should not";
  std::string comp = "[VariableMap mismatch] Global Variables have different values\nGlobal "
                     "Variable TestGlobal has values 10 and 12";
  EXPECT_TRUE(a.first == comp) << a.first;
}
} // anonymous namespace
