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
#include "dawn/Support/Casting.h"
#include "dawn/Support/STLExtras.h"
#include <gtest/gtest.h>

// Statements:
// -  Stmt
// -  BlockStmt
// -  ExprStmt
// -  ReturnStmt
// -  VarDeclStmt
// -  IfStmt
//
// Expressions
// -  Expr
// -  UnaryOperator
// -  BinaryOperator
// -  AssignmentExpr
// -  TernaryOperator
// -  FunCallExpr
// -  StencilFunCallExpr
// -  StencilFunArgExpr
// -  VarAccessExpr
// -  FieldAccessExpr
// -  LiteralAccessExpr

using namespace dawn;

namespace {

class ASTTest : public ::testing::Test {
protected:
  std::shared_ptr<BlockStmt> stmt_BlockStmt;
  std::shared_ptr<ExprStmt> stmt_ExprStmt;
  std::shared_ptr<ReturnStmt> stmt_ReturnStmt;
  std::shared_ptr<VarDeclStmt> stmt_VarDeclStmt;
  std::shared_ptr<IfStmt> stmt_IfStmt;

  std::shared_ptr<UnaryOperator> expr_UnaryOperator;
  std::shared_ptr<BinaryOperator> expr_BinaryOperator;
  std::shared_ptr<AssignmentExpr> expr_AssignmentExpr;
  std::shared_ptr<TernaryOperator> expr_TernaryOperator;
  std::shared_ptr<FunCallExpr> expr_FunCallExpr;
  std::shared_ptr<StencilFunCallExpr> expr_StencilFunCallExpr;
  std::shared_ptr<StencilFunArgExpr> expr_StencilFunArgExpr;
  std::shared_ptr<VarAccessExpr> expr_VarAccessExpr;
  std::shared_ptr<FieldAccessExpr> expr_FieldAccessExpr;
  std::shared_ptr<LiteralAccessExpr> expr_LiteralAccessExpr;

  void SetUp() override {
    //
    // Statements
    //
    stmt_ExprStmt = std::make_shared<ExprStmt>(std::make_shared<VarAccessExpr>("foo"));
    stmt_BlockStmt = std::make_shared<BlockStmt>(std::vector<std::shared_ptr<Stmt>>{
        std::make_shared<ExprStmt>(std::make_shared<VarAccessExpr>("foo"))});
    stmt_ReturnStmt = std::make_shared<ReturnStmt>(std::make_shared<VarAccessExpr>("foo"));
    stmt_VarDeclStmt = std::make_shared<VarDeclStmt>(
        Type(BuiltinTypeID::Float), "var", 0, "=",
        std::vector<std::shared_ptr<Expr>>{std::make_shared<VarAccessExpr>("foo")});
    stmt_IfStmt = std::make_shared<IfStmt>(
        std::make_shared<ExprStmt>(std::make_shared<VarAccessExpr>("foo1")),
        std::make_shared<ExprStmt>(std::make_shared<VarAccessExpr>("foo2")),
        std::make_shared<ExprStmt>(std::make_shared<VarAccessExpr>("foo3")));
    //
    // Expressions
    //
    expr_UnaryOperator =
        std::make_shared<UnaryOperator>(std::make_shared<VarAccessExpr>("foo"), "-");
    expr_BinaryOperator = std::make_shared<BinaryOperator>(
        std::make_shared<VarAccessExpr>("foo"), "+", std::make_shared<VarAccessExpr>("foo"));
    expr_AssignmentExpr = std::make_shared<AssignmentExpr>(std::make_shared<VarAccessExpr>("foo1"),
                                                           std::make_shared<VarAccessExpr>("foo2"));
    expr_TernaryOperator = std::make_shared<TernaryOperator>(
        std::make_shared<VarAccessExpr>("foo1"), std::make_shared<VarAccessExpr>("foo2"),
        std::make_shared<VarAccessExpr>("foo3"));
    expr_FunCallExpr = std::make_shared<FunCallExpr>("fun-call");
    expr_FunCallExpr->getArguments().push_back(std::make_shared<VarAccessExpr>("foo"));
    expr_StencilFunCallExpr = std::make_shared<StencilFunCallExpr>("stencil-fun-call");
    expr_StencilFunCallExpr->getArguments().push_back(std::make_shared<VarAccessExpr>("foo"));
    expr_StencilFunArgExpr = std::make_shared<StencilFunArgExpr>(1, 0, -1);
    expr_VarAccessExpr = std::make_shared<VarAccessExpr>("foo");
    expr_FieldAccessExpr = std::make_shared<FieldAccessExpr>(
        "foo", Array3i{1, 2, 3}, Array3i{-1, -1, -1}, Array3i{0, 0, 0}, false);
    expr_LiteralAccessExpr = std::make_shared<LiteralAccessExpr>("5.0", BuiltinTypeID::Float);
  }

  void TearDown() override {}
};

template <class T, class S>
std::shared_ptr<T> castAs(const std::shared_ptr<S>& ptr) {
  return std::static_pointer_cast<T>(ptr);
}

TEST_F(ASTTest, RTTI) {
  //
  // Statements
  //
  EXPECT_TRUE(isa<BlockStmt>(castAs<Stmt>(stmt_BlockStmt).get()));
  EXPECT_TRUE(isa<ExprStmt>(castAs<Stmt>(stmt_ExprStmt).get()));
  EXPECT_TRUE(isa<ReturnStmt>(castAs<Stmt>(stmt_ReturnStmt).get()));
  EXPECT_TRUE(isa<VarDeclStmt>(castAs<Stmt>(stmt_VarDeclStmt).get()));
  EXPECT_TRUE(isa<IfStmt>(castAs<Stmt>(stmt_IfStmt).get()));

  //
  // Expressions
  //
  EXPECT_TRUE(isa<UnaryOperator>(castAs<Expr>(expr_UnaryOperator).get()));
  EXPECT_TRUE(isa<BinaryOperator>(castAs<Expr>(expr_BinaryOperator).get()));
  EXPECT_TRUE(isa<AssignmentExpr>(castAs<Expr>(expr_AssignmentExpr).get()));
  EXPECT_TRUE(isa<TernaryOperator>(castAs<Expr>(expr_TernaryOperator).get()));
  EXPECT_TRUE(isa<FunCallExpr>(castAs<Expr>(expr_FunCallExpr).get()));
  EXPECT_TRUE(isa<StencilFunCallExpr>(castAs<Expr>(expr_StencilFunCallExpr).get()));
  EXPECT_TRUE(isa<StencilFunArgExpr>(castAs<Expr>(expr_StencilFunArgExpr).get()));
  EXPECT_TRUE(isa<VarAccessExpr>(castAs<Expr>(expr_VarAccessExpr).get()));
  EXPECT_TRUE(isa<FieldAccessExpr>(castAs<Expr>(expr_FieldAccessExpr).get()));
  EXPECT_TRUE(isa<LiteralAccessExpr>(castAs<Expr>(expr_LiteralAccessExpr).get()));
}

TEST_F(ASTTest, Clone) {
  //
  // Statements
  //
  auto clone_BlockStmt = castAs<Stmt>(stmt_BlockStmt)->clone();
  EXPECT_EQ(stmt_BlockStmt.use_count(), 1);
  EXPECT_EQ(clone_BlockStmt.use_count(), 1);
  EXPECT_EQ(*stmt_BlockStmt, *clone_BlockStmt);

  auto clone_ExprStmt = castAs<Stmt>(stmt_ExprStmt)->clone();
  EXPECT_EQ(stmt_ExprStmt.use_count(), 1);
  EXPECT_EQ(clone_ExprStmt.use_count(), 1);
  EXPECT_EQ(*stmt_ExprStmt, *clone_ExprStmt);

  auto clone_ReturnStmt = castAs<Stmt>(stmt_ReturnStmt)->clone();
  EXPECT_EQ(stmt_ReturnStmt.use_count(), 1);
  EXPECT_EQ(clone_ReturnStmt.use_count(), 1);
  EXPECT_EQ(*stmt_ReturnStmt, *clone_ReturnStmt);

  auto clone_VarDeclStmt = castAs<Stmt>(stmt_VarDeclStmt)->clone();
  EXPECT_EQ(stmt_VarDeclStmt.use_count(), 1);
  EXPECT_EQ(clone_VarDeclStmt.use_count(), 1);
  EXPECT_EQ(*stmt_VarDeclStmt, *clone_VarDeclStmt);

  auto clone_IfStmt = castAs<Stmt>(stmt_IfStmt)->clone();
  EXPECT_EQ(stmt_IfStmt.use_count(), 1);
  EXPECT_EQ(clone_IfStmt.use_count(), 1);
  EXPECT_EQ(*stmt_IfStmt, *clone_IfStmt);

  //
  // Expressions
  //
  auto clone_UnaryOperator = castAs<Expr>(expr_UnaryOperator)->clone();
  EXPECT_EQ(expr_UnaryOperator.use_count(), 1);
  EXPECT_EQ(clone_UnaryOperator.use_count(), 1);
  EXPECT_EQ(*expr_UnaryOperator, *clone_UnaryOperator);

  auto clone_BinaryOperator = castAs<Expr>(expr_BinaryOperator)->clone();
  EXPECT_EQ(expr_BinaryOperator.use_count(), 1);
  EXPECT_EQ(clone_BinaryOperator.use_count(), 1);
  EXPECT_EQ(*expr_BinaryOperator, *clone_BinaryOperator);

  auto clone_AssignmentExpr = castAs<Expr>(expr_AssignmentExpr)->clone();
  EXPECT_EQ(expr_AssignmentExpr.use_count(), 1);
  EXPECT_EQ(clone_AssignmentExpr.use_count(), 1);
  EXPECT_EQ(*expr_AssignmentExpr, *clone_AssignmentExpr);

  auto clone_TernaryOperator = castAs<Expr>(expr_TernaryOperator)->clone();
  EXPECT_EQ(expr_TernaryOperator.use_count(), 1);
  EXPECT_EQ(clone_TernaryOperator.use_count(), 1);
  EXPECT_EQ(*expr_TernaryOperator, *clone_TernaryOperator);

  auto clone_FunCallExpr = castAs<Expr>(expr_FunCallExpr)->clone();
  EXPECT_EQ(expr_FunCallExpr.use_count(), 1);
  EXPECT_EQ(clone_FunCallExpr.use_count(), 1);
  EXPECT_EQ(*expr_FunCallExpr, *clone_FunCallExpr);

  auto clone_StencilFunCallExpr = castAs<Expr>(expr_StencilFunCallExpr)->clone();
  EXPECT_EQ(expr_StencilFunCallExpr.use_count(), 1);
  EXPECT_EQ(clone_StencilFunCallExpr.use_count(), 1);
  EXPECT_EQ(*expr_StencilFunCallExpr, *clone_StencilFunCallExpr);

  auto clone_StencilFunArgExpr = castAs<Expr>(expr_StencilFunArgExpr)->clone();
  EXPECT_EQ(expr_StencilFunArgExpr.use_count(), 1);
  EXPECT_EQ(clone_StencilFunArgExpr.use_count(), 1);
  EXPECT_EQ(*expr_StencilFunArgExpr, *clone_StencilFunArgExpr);

  auto clone_VarAccessExpr = castAs<Expr>(expr_VarAccessExpr)->clone();
  EXPECT_EQ(expr_VarAccessExpr.use_count(), 1);
  EXPECT_EQ(clone_VarAccessExpr.use_count(), 1);
  EXPECT_EQ(*expr_VarAccessExpr, *clone_VarAccessExpr);

  auto clone_FieldAccessExpr = castAs<Expr>(expr_FieldAccessExpr)->clone();
  EXPECT_EQ(expr_FieldAccessExpr.use_count(), 1);
  EXPECT_EQ(clone_FieldAccessExpr.use_count(), 1);
  EXPECT_EQ(*expr_FieldAccessExpr, *clone_FieldAccessExpr);

  auto clone_LiteralAccessExpr = castAs<Expr>(expr_LiteralAccessExpr)->clone();
  EXPECT_EQ(expr_LiteralAccessExpr.use_count(), 1);
  EXPECT_EQ(clone_LiteralAccessExpr.use_count(), 1);
  EXPECT_EQ(*expr_LiteralAccessExpr, *clone_LiteralAccessExpr);

  // Not the same type
  EXPECT_NE(*expr_FieldAccessExpr, *expr_LiteralAccessExpr);
}

TEST_F(ASTTest, ReplaceExpr) {
  auto foo_var = std::make_shared<VarAccessExpr>("foo");

  replaceOldExprWithNewExprInStmt(stmt_BlockStmt,
                    dyn_cast<ExprStmt>(stmt_BlockStmt->getStatements().front().get())->getExpr(),
                    foo_var);
  EXPECT_TRUE(dyn_cast<ExprStmt>(stmt_BlockStmt->getStatements().front().get())
                  ->getExpr()
                  ->equals(foo_var));

  replaceOldExprWithNewExprInStmt(stmt_ExprStmt, stmt_ExprStmt->getExpr(), foo_var);
  EXPECT_TRUE(stmt_ExprStmt->getExpr()->equals(foo_var));

  replaceOldExprWithNewExprInStmt(stmt_ReturnStmt, stmt_ReturnStmt->getExpr(), foo_var);
  EXPECT_TRUE(stmt_ReturnStmt->getExpr()->equals(foo_var));

  replaceOldExprWithNewExprInStmt(stmt_VarDeclStmt, stmt_VarDeclStmt->getInitList().front(), foo_var);
  EXPECT_TRUE(stmt_VarDeclStmt->getInitList().front()->equals(foo_var));

  replaceOldExprWithNewExprInStmt(stmt_IfStmt, stmt_IfStmt->getCondExpr(), foo_var);
  EXPECT_TRUE(stmt_IfStmt->getCondExpr()->equals(foo_var));
}

TEST_F(ASTTest, EvalExprBoolean) {
  {
    auto expr = std::make_shared<LiteralAccessExpr>("0", BuiltinTypeID::Boolean);
    bool res;
    EXPECT_TRUE(evalExprAsBoolean(expr, res));
    EXPECT_EQ(res, false);
  }

  {
    auto expr = std::make_shared<LiteralAccessExpr>("1", BuiltinTypeID::Boolean);
    bool res;
    EXPECT_TRUE(evalExprAsBoolean(expr, res));
    EXPECT_EQ(res, true);
  }

  {
    auto expr = std::make_shared<LiteralAccessExpr>("true", BuiltinTypeID::Boolean);
    bool res;
    EXPECT_TRUE(evalExprAsBoolean(expr, res));
    EXPECT_EQ(res, true);
  }

  {
    auto expr = std::make_shared<LiteralAccessExpr>("false", BuiltinTypeID::Boolean);
    bool res;
    EXPECT_TRUE(evalExprAsBoolean(expr, res));
    EXPECT_EQ(res, false);
  }

  {
    auto expr = std::make_shared<LiteralAccessExpr>("2", BuiltinTypeID::Integer);
    bool res;
    EXPECT_TRUE(evalExprAsBoolean(expr, res));
    EXPECT_EQ(res, true);
  }

  {
    auto expr = std::make_shared<LiteralAccessExpr>("2.2", BuiltinTypeID::Float);
    bool res;
    EXPECT_TRUE(evalExprAsBoolean(expr, res));
    EXPECT_EQ(res, true);
  }

  {
    auto expr = std::make_shared<BinaryOperator>(
        std::make_shared<LiteralAccessExpr>("true", BuiltinTypeID::Boolean), "&&",
        std::make_shared<LiteralAccessExpr>("false", BuiltinTypeID::Boolean));
    bool res;
    EXPECT_TRUE(evalExprAsBoolean(expr, res));
    EXPECT_EQ(res, false);
  }

  {
    auto expr = std::make_shared<BinaryOperator>(
        std::make_shared<LiteralAccessExpr>("true", BuiltinTypeID::Boolean), "||",
        std::make_shared<LiteralAccessExpr>("false", BuiltinTypeID::Boolean));
    bool res;
    EXPECT_TRUE(evalExprAsBoolean(expr, res));
    EXPECT_EQ(res, true);
  }

  {
    auto expr = std::make_shared<BinaryOperator>(
        std::make_shared<LiteralAccessExpr>("true", BuiltinTypeID::Boolean), "==",
        std::make_shared<LiteralAccessExpr>("false", BuiltinTypeID::Boolean));
    bool res;
    EXPECT_TRUE(evalExprAsBoolean(expr, res));
    EXPECT_EQ(res, false);
  }

  {
    auto expr = std::make_shared<BinaryOperator>(
        std::make_shared<LiteralAccessExpr>("true", BuiltinTypeID::Boolean), "!=",
        std::make_shared<LiteralAccessExpr>("false", BuiltinTypeID::Boolean));
    bool res;
    EXPECT_TRUE(evalExprAsBoolean(expr, res));
    EXPECT_EQ(res, true);
  }

  {
    auto expr = std::make_shared<TernaryOperator>(
        std::make_shared<BinaryOperator>(
            std::make_shared<LiteralAccessExpr>("true", BuiltinTypeID::Boolean), "&&",
            std::make_shared<LiteralAccessExpr>("true", BuiltinTypeID::Boolean)),
        std::make_shared<LiteralAccessExpr>("true", BuiltinTypeID::Boolean),
        std::make_shared<LiteralAccessExpr>("false", BuiltinTypeID::Boolean));
    bool res;
    EXPECT_TRUE(evalExprAsBoolean(expr, res));
    EXPECT_EQ(res, true);
  }

  {
    auto expr = std::make_shared<TernaryOperator>(
        std::make_shared<BinaryOperator>(
            std::make_shared<LiteralAccessExpr>("true", BuiltinTypeID::Boolean), "&&",
            std::make_shared<LiteralAccessExpr>("false", BuiltinTypeID::Boolean)),
        std::make_shared<LiteralAccessExpr>("true", BuiltinTypeID::Boolean),
        std::make_shared<LiteralAccessExpr>("false", BuiltinTypeID::Boolean));
    bool res;
    EXPECT_TRUE(evalExprAsBoolean(expr, res));
    EXPECT_EQ(res, false);
  }

  {
    auto expr = std::make_shared<BinaryOperator>(
        std::make_shared<LiteralAccessExpr>("1", BuiltinTypeID::Integer), ">",
        std::make_shared<LiteralAccessExpr>("2", BuiltinTypeID::Integer));
    bool res;
    EXPECT_TRUE(evalExprAsBoolean(expr, res));
    EXPECT_EQ(res, false);
  }

  {
    auto expr = std::make_shared<BinaryOperator>(
        std::make_shared<LiteralAccessExpr>("1", BuiltinTypeID::Integer), "<",
        std::make_shared<LiteralAccessExpr>("2", BuiltinTypeID::Integer));
    bool res;
    EXPECT_TRUE(evalExprAsBoolean(expr, res));
    EXPECT_EQ(res, true);
  }

  {
    auto expr = std::make_shared<BinaryOperator>(
        std::make_shared<LiteralAccessExpr>("1", BuiltinTypeID::Integer), ">=",
        std::make_shared<LiteralAccessExpr>("2", BuiltinTypeID::Integer));
    bool res;
    EXPECT_TRUE(evalExprAsBoolean(expr, res));
    EXPECT_EQ(res, false);
  }

  {
    auto expr = std::make_shared<BinaryOperator>(
        std::make_shared<LiteralAccessExpr>("1", BuiltinTypeID::Integer), "<=",
        std::make_shared<LiteralAccessExpr>("2", BuiltinTypeID::Integer));
    bool res;
    EXPECT_TRUE(evalExprAsBoolean(expr, res));
    EXPECT_EQ(res, true);
  }
}

TEST_F(ASTTest, EvalExprInteger) {
  {
    auto expr = std::make_shared<LiteralAccessExpr>("0", BuiltinTypeID::Integer);
    int res;
    EXPECT_TRUE(evalExprAsInteger(expr, res));
    EXPECT_EQ(res, 0);
  }

  {
    auto expr = std::make_shared<LiteralAccessExpr>("1", BuiltinTypeID::Integer);
    int res;
    EXPECT_TRUE(evalExprAsInteger(expr, res));
    EXPECT_EQ(res, 1);
  }

  {
    auto expr = std::make_shared<LiteralAccessExpr>("2", BuiltinTypeID::Integer);
    int res;
    EXPECT_TRUE(evalExprAsInteger(expr, res));
    EXPECT_EQ(res, 2);
  }

  {
    auto expr = std::make_shared<LiteralAccessExpr>("2.2", BuiltinTypeID::Float);
    int res;
    EXPECT_TRUE(evalExprAsInteger(expr, res));
    EXPECT_EQ(res, 2);
  }

  {
    auto expr = std::make_shared<LiteralAccessExpr>("true", BuiltinTypeID::Boolean);
    int res;
    EXPECT_TRUE(evalExprAsInteger(expr, res));
    EXPECT_EQ(res, 1);
  }

  {
    auto expr = std::make_shared<BinaryOperator>(
        std::make_shared<LiteralAccessExpr>("1", BuiltinTypeID::Integer), "+",
        std::make_shared<LiteralAccessExpr>("2", BuiltinTypeID::Integer));
    int res;
    EXPECT_TRUE(evalExprAsInteger(expr, res));
    EXPECT_EQ(res, 3);
  }

  {
    auto expr = std::make_shared<BinaryOperator>(
        std::make_shared<LiteralAccessExpr>("1", BuiltinTypeID::Integer), "-",
        std::make_shared<LiteralAccessExpr>("2", BuiltinTypeID::Integer));
    int res;
    EXPECT_TRUE(evalExprAsInteger(expr, res));
    EXPECT_EQ(res, -1);
  }

  {
    auto expr = std::make_shared<BinaryOperator>(
        std::make_shared<LiteralAccessExpr>("1", BuiltinTypeID::Integer), "*",
        std::make_shared<LiteralAccessExpr>("2", BuiltinTypeID::Integer));
    int res;
    EXPECT_TRUE(evalExprAsInteger(expr, res));
    EXPECT_EQ(res, 2);
  }

  {
    auto expr = std::make_shared<BinaryOperator>(
        std::make_shared<LiteralAccessExpr>("1", BuiltinTypeID::Integer), "/",
        std::make_shared<LiteralAccessExpr>("2", BuiltinTypeID::Integer));
    int res;
    EXPECT_TRUE(evalExprAsInteger(expr, res));
    EXPECT_EQ(res, 0);
  }
}

TEST_F(ASTTest, EvalExprFloat) {
  {
    auto expr = std::make_shared<LiteralAccessExpr>("0", BuiltinTypeID::Float);
    double res;
    EXPECT_TRUE(evalExprAsDouble(expr, res));
    EXPECT_EQ(res, 0.0);
  }

  {
    auto expr = std::make_shared<LiteralAccessExpr>("0.5", BuiltinTypeID::Float);
    double res;
    EXPECT_TRUE(evalExprAsDouble(expr, res));
    EXPECT_EQ(res, 0.5);
  }

  {
    auto expr = std::make_shared<LiteralAccessExpr>("1", BuiltinTypeID::Integer);
    double res;
    EXPECT_TRUE(evalExprAsDouble(expr, res));
    EXPECT_EQ(res, 1.0);
  }

  {
    auto expr = std::make_shared<LiteralAccessExpr>("true", BuiltinTypeID::Boolean);
    double res;
    EXPECT_TRUE(evalExprAsDouble(expr, res));
    EXPECT_EQ(res, 1.0);
  }

  {
    auto expr = std::make_shared<BinaryOperator>(
        std::make_shared<LiteralAccessExpr>("1.4", BuiltinTypeID::Float), "+",
        std::make_shared<LiteralAccessExpr>("2.5", BuiltinTypeID::Float));
    double res;
    EXPECT_TRUE(evalExprAsDouble(expr, res));
    EXPECT_FLOAT_EQ(res, 3.9);
  }

  {
    auto expr = std::make_shared<BinaryOperator>(
        std::make_shared<LiteralAccessExpr>("1.4", BuiltinTypeID::Float), "-",
        std::make_shared<LiteralAccessExpr>("2.5", BuiltinTypeID::Float));
    double res;
    EXPECT_TRUE(evalExprAsDouble(expr, res));
    EXPECT_FLOAT_EQ(res, -1.1);
  }

  {
    auto expr = std::make_shared<BinaryOperator>(
        std::make_shared<LiteralAccessExpr>("1.4", BuiltinTypeID::Float), "*",
        std::make_shared<LiteralAccessExpr>("2.5", BuiltinTypeID::Float));
    double res;
    EXPECT_TRUE(evalExprAsDouble(expr, res));
    EXPECT_FLOAT_EQ(res, 3.5);
  }

  {
    auto expr = std::make_shared<BinaryOperator>(
        std::make_shared<LiteralAccessExpr>("1.4", BuiltinTypeID::Float), "/",
        std::make_shared<LiteralAccessExpr>("2.5", BuiltinTypeID::Float));
    double res;
    EXPECT_TRUE(evalExprAsDouble(expr, res));
    EXPECT_FLOAT_EQ(res, 0.56);
  }
}

TEST_F(ASTTest, EvalExprWithVariables) {

  {
    std::unordered_map<std::string, double> variableMap = {{"foo", 0.0}};

    auto expr = std::make_shared<BinaryOperator>(
        std::make_shared<LiteralAccessExpr>("true", BuiltinTypeID::Boolean), "&&",
        std::make_shared<VarAccessExpr>("foo"));
    bool res;
    EXPECT_TRUE(evalExprAsBoolean(expr, res, variableMap));
    EXPECT_EQ(res, false);
  }

  {
    std::unordered_map<std::string, double> variableMap = {{"foo", 1.0}};

    auto expr = std::make_shared<BinaryOperator>(
        std::make_shared<LiteralAccessExpr>("true", BuiltinTypeID::Boolean), "&&",
        std::make_shared<VarAccessExpr>("foo"));
    bool res;
    EXPECT_TRUE(evalExprAsBoolean(expr, res, variableMap));
    EXPECT_EQ(res, true);
  }

  {
    std::unordered_map<std::string, double> variableMap = {{"foo", 3.0}};

    auto expr = std::make_shared<BinaryOperator>(
        std::make_shared<LiteralAccessExpr>("5", BuiltinTypeID::Float), ">",
        std::make_shared<VarAccessExpr>("foo"));
    bool res;
    EXPECT_TRUE(evalExprAsBoolean(expr, res, variableMap));
    EXPECT_EQ(res, true);
  }
}

} // anonymous namespace
