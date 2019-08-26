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

class ReplaceVisitor : public sir::ASTVisitorPostOrder, public NonCopyable {
protected:
public:
  ReplaceVisitor() {}

  virtual ~ReplaceVisitor() {}

  virtual std::shared_ptr<sir::Expr>
  postVisitNode(std::shared_ptr<sir::FieldAccessExpr> const& expr) override {
    return std::make_shared<sir::FieldAccessExpr>(expr->getName() + "post");
  }

  virtual std::shared_ptr<sir::Expr>
  postVisitNode(std::shared_ptr<sir::LiteralAccessExpr> const& expr) override {
    return std::make_shared<sir::LiteralAccessExpr>(
        std::to_string((std::stof(expr->getValue()) + 2) * 7), BuiltinTypeID::Integer);
  }

  virtual std::shared_ptr<sir::Expr>
  postVisitNode(std::shared_ptr<sir::VarAccessExpr> const& expr) override {
    return std::make_shared<sir::VarAccessExpr>(
        expr->getName(), std::make_shared<sir::LiteralAccessExpr>("99", BuiltinTypeID::Integer));
  }
};

class CheckVisitor : public sir::ASTVisitorForwarding, public NonCopyable {

  bool result_ = true;

public:
  CheckVisitor() {}

  bool result() const { return result_; }
  virtual ~CheckVisitor() {}

  virtual void visit(std::shared_ptr<sir::FieldAccessExpr> const& expr) override {
    std::regex self_regex("post", std::regex_constants::basic);
    result_ = result_ && (std::regex_search(expr->getName(), self_regex));
  }

  virtual void visit(std::shared_ptr<sir::LiteralAccessExpr> const& expr) override {
    double val = std::stof(expr->getValue()) / (double)7.0 - 2;
    if(val != 0 && val != 1 && std::fabs((val - 3.14)) / 3.14 > 1e-6) {
      result_ = false;
    }
  }

  virtual void visit(std::shared_ptr<sir::VarAccessExpr> const& expr) override {
    DAWN_ASSERT(isa<sir::LiteralAccessExpr>(*(expr->getIndex())));
    result_ =
        result_ &&
        (std::dynamic_pointer_cast<sir::LiteralAccessExpr>(expr->getIndex())->getValue() == "99");
  }
};
class ASTPostOrderVisitor : public ::testing::Test {
protected:
  std::shared_ptr<sir::BlockStmt> blockStmt_;

  void SetUp() override {
    blockStmt_ = std::make_shared<sir::BlockStmt>();

    // pi = {3.14, 3.14, 3.14};
    Type floatType(BuiltinTypeID::Float);
    std::shared_ptr<sir::LiteralAccessExpr> pi =
        std::make_shared<sir::LiteralAccessExpr>("3.14", BuiltinTypeID::Float);
    std::vector<std::shared_ptr<sir::Expr>> initList = {pi, pi, pi};
    std::shared_ptr<sir::VarDeclStmt> varDcl =
        std::make_shared<sir::VarDeclStmt>(floatType, "u1", 3, "=", initList);

    blockStmt_->push_back(varDcl);

    // index to access pi[1]
    std::shared_ptr<sir::LiteralAccessExpr> index0 =
        std::make_shared<sir::LiteralAccessExpr>("0", BuiltinTypeID::Integer);
    std::shared_ptr<sir::LiteralAccessExpr> index1 =
        std::make_shared<sir::LiteralAccessExpr>("1", BuiltinTypeID::Integer);

    // f1 = f2+pi[1]
    std::shared_ptr<sir::FieldAccessExpr> f1 = std::make_shared<sir::FieldAccessExpr>("f1");
    std::shared_ptr<sir::FieldAccessExpr> f2 = std::make_shared<sir::FieldAccessExpr>("f2");
    std::shared_ptr<sir::VarAccessExpr> varAccess =
        std::make_shared<sir::VarAccessExpr>("pi", index1);
    std::shared_ptr<sir::BinaryOperator> rightExpr =
        std::make_shared<sir::BinaryOperator>(f2, "+", varAccess);
    std::shared_ptr<sir::AssignmentExpr> assignExpr =
        std::make_shared<sir::AssignmentExpr>(f1, rightExpr);
    std::shared_ptr<sir::ExprStmt> exprStmt1 = std::make_shared<sir::ExprStmt>(assignExpr);

    blockStmt_->push_back(exprStmt1);

    std::shared_ptr<sir::VarAccessExpr> varAccess1 =
        std::make_shared<sir::VarAccessExpr>("pi", index0);

    std::shared_ptr<sir::LiteralAccessExpr> val0 =
        std::make_shared<sir::LiteralAccessExpr>("0", BuiltinTypeID::Float);

    std::shared_ptr<sir::BinaryOperator> equal =
        std::make_shared<sir::BinaryOperator>(varAccess1, "==", val0);
    std::shared_ptr<sir::ExprStmt> condStmt = std::make_shared<sir::ExprStmt>(equal);
    std::shared_ptr<sir::ReturnStmt> returnIf = std::make_shared<sir::ReturnStmt>(f1);
    std::shared_ptr<sir::ReturnStmt> returnElse = std::make_shared<sir::ReturnStmt>(f2);

    std::shared_ptr<sir::IfStmt> ifStmt =
        std::make_shared<sir::IfStmt>(condStmt, returnIf, returnElse);
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

class ReplaceAssignVisitor : public sir::ASTVisitorPostOrder, public NonCopyable {
protected:
public:
  ReplaceAssignVisitor() {}

  virtual ~ReplaceAssignVisitor() {}

  virtual std::shared_ptr<sir::Expr>
  postVisitNode(std::shared_ptr<sir::AssignmentExpr> const& expr) override {
    return std::make_shared<sir::AssignmentExpr>(std::make_shared<sir::FieldAccessExpr>("demo_out"),
                                                 std::make_shared<sir::FieldAccessExpr>("demo_in"));
  }
};

class CheckAssignVisitor : public sir::ASTVisitorForwarding, public NonCopyable {

  bool result_ = true;

public:
  CheckAssignVisitor() {}

  bool result() const { return result_; }
  virtual ~CheckAssignVisitor() {}

  virtual void visit(std::shared_ptr<sir::AssignmentExpr> const& expr) override {
    DAWN_ASSERT(isa<sir::FieldAccessExpr>(*(expr->getLeft())));

    result_ =
        result_ &&
        (std::dynamic_pointer_cast<sir::FieldAccessExpr>(expr->getLeft())->getName() == "demo_out");
    result_ =
        result_ &&
        (std::dynamic_pointer_cast<sir::FieldAccessExpr>(expr->getRight())->getName() == "demo_in");
  }
};

TEST_F(ASTPostOrderVisitor, assignment) {

  ReplaceAssignVisitor repl;
  blockStmt_->acceptAndReplace(repl);

  CheckAssignVisitor checker;
  blockStmt_->accept(checker);

  ASSERT_TRUE(checker.result());
}

class ReplaceIfVisitor : public sir::ASTVisitorPostOrder, public NonCopyable {
protected:
public:
  ReplaceIfVisitor() {}

  virtual ~ReplaceIfVisitor() {}

  virtual std::shared_ptr<sir::Stmt>
  postVisitNode(std::shared_ptr<sir::IfStmt> const& stmt) override {
    DAWN_ASSERT(isa<sir::BinaryOperator>(*(stmt->getCondExpr())));

    std::shared_ptr<sir::BinaryOperator> op =
        std::dynamic_pointer_cast<sir::BinaryOperator>(stmt->getCondExpr());

    std::shared_ptr<sir::ExprStmt> gt = std::make_shared<sir::ExprStmt>(
        std::make_shared<sir::BinaryOperator>(op->getLeft(), ">=", op->getRight()));

    std::shared_ptr<sir::ExprStmt> ifStmt =
        std::make_shared<sir::ExprStmt>(std::make_shared<sir::NOPExpr>());
    std::shared_ptr<sir::ExprStmt> elseStmt =
        std::make_shared<sir::ExprStmt>(std::make_shared<sir::NOPExpr>());

    return std::make_shared<sir::IfStmt>(gt, ifStmt, elseStmt);
  }
};

class CheckIfVisitor : public sir::ASTVisitorForwarding, public NonCopyable {

  bool result_ = true;

public:
  CheckIfVisitor() {}

  bool result() const { return result_; }
  virtual ~CheckIfVisitor() {}

  virtual void visit(std::shared_ptr<sir::IfStmt> const& stmt) override {
    DAWN_ASSERT(isa<sir::BinaryOperator>(*(stmt->getCondExpr())));

    std::shared_ptr<sir::BinaryOperator> op =
        std::dynamic_pointer_cast<sir::BinaryOperator>(stmt->getCondExpr());

    result_ = result_ && (std::string(op->getOp()) == ">=");

    DAWN_ASSERT(isa<sir::ExprStmt>(*(stmt->getThenStmt())));
    std::shared_ptr<sir::ExprStmt> thenExpr =
        std::dynamic_pointer_cast<sir::ExprStmt>(stmt->getThenStmt());
    result_ = result_ && isa<sir::NOPExpr>(*(thenExpr->getExpr()));
    std::shared_ptr<sir::ExprStmt> elseExpr =
        std::dynamic_pointer_cast<sir::ExprStmt>(stmt->getElseStmt());
    result_ = result_ && isa<sir::NOPExpr>(*(elseExpr->getExpr()));
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
