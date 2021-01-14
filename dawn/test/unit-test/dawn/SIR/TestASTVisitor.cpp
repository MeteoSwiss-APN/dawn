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
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/STLExtras.h"
#include <cmath>
#include <gtest/gtest.h>
#include <regex>

using namespace dawn;

namespace {

class ReplaceVisitor : public ast::ASTVisitorPostOrder, public NonCopyable {
protected:
public:
  ReplaceVisitor() {}

  virtual ~ReplaceVisitor() {}

  virtual std::shared_ptr<ast::Expr>
  postVisitNode(std::shared_ptr<ast::FieldAccessExpr> const& expr) override {
    return std::make_shared<ast::FieldAccessExpr>(expr->getName() + "post");
  }

  virtual std::shared_ptr<ast::Expr>
  postVisitNode(std::shared_ptr<ast::LiteralAccessExpr> const& expr) override {
    return std::make_shared<ast::LiteralAccessExpr>(
        std::to_string((std::stof(expr->getValue()) + 2) * 7), BuiltinTypeID::Integer);
  }

  virtual std::shared_ptr<ast::Expr>
  postVisitNode(std::shared_ptr<ast::VarAccessExpr> const& expr) override {
    return std::make_shared<ast::VarAccessExpr>(
        expr->getName(), std::make_shared<ast::LiteralAccessExpr>("99", BuiltinTypeID::Integer));
  }
};

class CheckVisitor : public ast::ASTVisitorForwarding, public NonCopyable {

  bool result_ = true;

public:
  CheckVisitor() {}

  bool result() const { return result_; }
  virtual ~CheckVisitor() {}

  virtual void visit(std::shared_ptr<ast::FieldAccessExpr> const& expr) override {
    std::regex self_regex("post", std::regex_constants::basic);
    result_ = result_ && (std::regex_search(expr->getName(), self_regex));
  }

  virtual void visit(std::shared_ptr<ast::LiteralAccessExpr> const& expr) override {
    double val = std::stof(expr->getValue()) / (double)7.0 - 2;
    if(val != 0 && val != 1 && std::fabs((val - 3.14)) / 3.14 > 1e-6) {
      result_ = false;
    }
  }

  virtual void visit(std::shared_ptr<ast::VarAccessExpr> const& expr) override {
    DAWN_ASSERT(isa<ast::LiteralAccessExpr>(*(expr->getIndex())));
    result_ =
        result_ &&
        (std::dynamic_pointer_cast<ast::LiteralAccessExpr>(expr->getIndex())->getValue() == "99");
  }
};
class ASTPostOrderVisitor : public ::testing::Test {
protected:
  std::shared_ptr<ast::BlockStmt> blockStmt_;

  void SetUp() override {
    blockStmt_ = sir::makeBlockStmt();

    // pi = {3.14, 3.14, 3.14};
    Type floatType(BuiltinTypeID::Float);
    std::shared_ptr<ast::LiteralAccessExpr> pi =
        std::make_shared<ast::LiteralAccessExpr>("3.14", BuiltinTypeID::Float);
    std::vector<std::shared_ptr<ast::Expr>> initList = {pi, pi, pi};
    std::shared_ptr<ast::VarDeclStmt> varDcl =
        sir::makeVarDeclStmt(floatType, "u1", 3, "=", initList);

    blockStmt_->push_back(varDcl);

    // index to access pi[1]
    std::shared_ptr<ast::LiteralAccessExpr> index0 =
        std::make_shared<ast::LiteralAccessExpr>("0", BuiltinTypeID::Integer);
    std::shared_ptr<ast::LiteralAccessExpr> index1 =
        std::make_shared<ast::LiteralAccessExpr>("1", BuiltinTypeID::Integer);

    // f1 = f2+pi[1]
    std::shared_ptr<ast::FieldAccessExpr> f1 = std::make_shared<ast::FieldAccessExpr>("f1");
    std::shared_ptr<ast::FieldAccessExpr> f2 = std::make_shared<ast::FieldAccessExpr>("f2");
    std::shared_ptr<ast::VarAccessExpr> varAccess =
        std::make_shared<ast::VarAccessExpr>("pi", index1);
    std::shared_ptr<ast::BinaryOperator> rightExpr =
        std::make_shared<ast::BinaryOperator>(f2, "+", varAccess);
    std::shared_ptr<ast::AssignmentExpr> assignExpr =
        std::make_shared<ast::AssignmentExpr>(f1, rightExpr);
    std::shared_ptr<ast::ExprStmt> exprStmt1 = sir::makeExprStmt(assignExpr);

    blockStmt_->push_back(exprStmt1);

    std::shared_ptr<ast::VarAccessExpr> varAccess1 =
        std::make_shared<ast::VarAccessExpr>("pi", index0);

    std::shared_ptr<ast::LiteralAccessExpr> val0 =
        std::make_shared<ast::LiteralAccessExpr>("0", BuiltinTypeID::Float);

    std::shared_ptr<ast::BinaryOperator> equal =
        std::make_shared<ast::BinaryOperator>(varAccess1, "==", val0);
    std::shared_ptr<ast::ExprStmt> condStmt = sir::makeExprStmt(equal);
    std::shared_ptr<ast::ReturnStmt> returnIf = sir::makeReturnStmt(f1);
    std::shared_ptr<ast::ReturnStmt> returnElse = sir::makeReturnStmt(f2);

    std::shared_ptr<ast::IfStmt> ifStmt = sir::makeIfStmt(condStmt, returnIf, returnElse);
    blockStmt_->push_back(ifStmt);
  }
};

TEST_F(ASTPostOrderVisitor, accesors) {

  ReplaceVisitor repl;
  blockStmt_->acceptAndReplace(repl);

  CheckVisitor checker;
  blockStmt_->accept(checker);

  ASSERT_TRUE(checker.result());
}

class ReplaceAssignVisitor : public ast::ASTVisitorPostOrder, public NonCopyable {
protected:
public:
  ReplaceAssignVisitor() {}

  virtual ~ReplaceAssignVisitor() {}

  virtual std::shared_ptr<ast::Expr>
  postVisitNode(std::shared_ptr<ast::AssignmentExpr> const& expr) override {
    return std::make_shared<ast::AssignmentExpr>(std::make_shared<ast::FieldAccessExpr>("demo_out"),
                                                 std::make_shared<ast::FieldAccessExpr>("demo_in"));
  }
};

class CheckAssignVisitor : public ast::ASTVisitorForwarding, public NonCopyable {

  bool result_ = true;

public:
  CheckAssignVisitor() {}

  bool result() const { return result_; }
  virtual ~CheckAssignVisitor() {}

  virtual void visit(std::shared_ptr<ast::AssignmentExpr> const& expr) override {
    DAWN_ASSERT(isa<ast::FieldAccessExpr>(*(expr->getLeft())));

    result_ =
        result_ &&
        (std::dynamic_pointer_cast<ast::FieldAccessExpr>(expr->getLeft())->getName() == "demo_out");
    result_ =
        result_ &&
        (std::dynamic_pointer_cast<ast::FieldAccessExpr>(expr->getRight())->getName() == "demo_in");
  }
};

TEST_F(ASTPostOrderVisitor, assignment) {

  ReplaceAssignVisitor repl;
  blockStmt_->acceptAndReplace(repl);

  CheckAssignVisitor checker;
  blockStmt_->accept(checker);

  ASSERT_TRUE(checker.result());
}

class ReplaceIfVisitor : public ast::ASTVisitorPostOrder, public NonCopyable {
protected:
public:
  ReplaceIfVisitor() {}

  virtual ~ReplaceIfVisitor() {}

  virtual std::shared_ptr<ast::Stmt>
  postVisitNode(std::shared_ptr<ast::IfStmt> const& stmt) override {
    DAWN_ASSERT(isa<ast::BinaryOperator>(*(stmt->getCondExpr())));

    std::shared_ptr<ast::BinaryOperator> op =
        std::dynamic_pointer_cast<ast::BinaryOperator>(stmt->getCondExpr());

    std::shared_ptr<ast::ExprStmt> gt = sir::makeExprStmt(
        std::make_shared<ast::BinaryOperator>(op->getLeft(), ">=", op->getRight()));

    std::shared_ptr<ast::ExprStmt> ifStmt = sir::makeExprStmt(std::make_shared<ast::NOPExpr>());
    std::shared_ptr<ast::ExprStmt> elseStmt = sir::makeExprStmt(std::make_shared<ast::NOPExpr>());

    return sir::makeIfStmt(gt, ifStmt, elseStmt);
  }
};

class CheckIfVisitor : public ast::ASTVisitorForwarding, public NonCopyable {

  bool result_ = true;

public:
  CheckIfVisitor() {}

  bool result() const { return result_; }
  virtual ~CheckIfVisitor() {}

  virtual void visit(std::shared_ptr<ast::IfStmt> const& stmt) override {
    DAWN_ASSERT(isa<ast::BinaryOperator>(*(stmt->getCondExpr())));

    std::shared_ptr<ast::BinaryOperator> op =
        std::dynamic_pointer_cast<ast::BinaryOperator>(stmt->getCondExpr());

    result_ = result_ && (op->getOp() == ">=");

    DAWN_ASSERT(isa<ast::ExprStmt>(*(stmt->getThenStmt())));
    std::shared_ptr<ast::ExprStmt> thenExpr =
        std::dynamic_pointer_cast<ast::ExprStmt>(stmt->getThenStmt());
    result_ = result_ && isa<ast::NOPExpr>(*(thenExpr->getExpr()));
    std::shared_ptr<ast::ExprStmt> elseExpr =
        std::dynamic_pointer_cast<ast::ExprStmt>(stmt->getElseStmt());
    result_ = result_ && isa<ast::NOPExpr>(*(elseExpr->getExpr()));
  }
};

TEST_F(ASTPostOrderVisitor, ifblock) {

  ReplaceIfVisitor repl;
  blockStmt_->acceptAndReplace(repl);

  CheckIfVisitor checker;
  blockStmt_->accept(checker);

  ASSERT_TRUE(checker.result());
}

} // anonymous namespace
