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
#include <cmath>
#include <gtest/gtest.h>
#include <regex>

using namespace dawn;

namespace {

class ReplaceVisitor : public ASTVisitorPostOrder, public NonCopyable {
protected:
public:
  ReplaceVisitor() {}

  virtual ~ReplaceVisitor() {}

  virtual std::shared_ptr<Expr>
  postVisitNode(std::shared_ptr<FieldAccessExpr> const& expr) override {
    return std::make_shared<FieldAccessExpr>(expr->getName() + "post");
  }

  virtual std::shared_ptr<Expr>
  postVisitNode(std::shared_ptr<LiteralAccessExpr> const& expr) override {
    return std::make_shared<LiteralAccessExpr>(
        std::to_string((std::stof(expr->getValue()) + 2) * 7), BuiltinTypeID::Integer);
  }

  virtual std::shared_ptr<Expr> postVisitNode(std::shared_ptr<VarAccessExpr> const& expr) override {
    return std::make_shared<VarAccessExpr>(
        expr->getName(), std::make_shared<LiteralAccessExpr>("99", BuiltinTypeID::Integer));
  }
};

class CheckVisitor : public ASTVisitorForwarding, public NonCopyable {

  bool result_ = true;

public:
  CheckVisitor() {}

  bool result() const { return result_; }
  virtual ~CheckVisitor() {}

  virtual void visit(std::shared_ptr<FieldAccessExpr> const& expr) override {
    std::regex self_regex("post", std::regex_constants::basic);
    result_ = result_ && (std::regex_search(expr->getName(), self_regex));
  }

  virtual void visit(std::shared_ptr<LiteralAccessExpr> const& expr) override {
    double val = std::stof(expr->getValue()) / (double)7.0 - 2;
    if(val != 0 && val != 1 && std::fabs((val - 3.14)) / 3.14 > 1e-6) {
      result_ = false;
    }
  }

  virtual void visit(std::shared_ptr<VarAccessExpr> const& expr) override {
    DAWN_ASSERT(isa<LiteralAccessExpr>(*(expr->getIndex())));
    result_ = result_ &&
              (std::dynamic_pointer_cast<LiteralAccessExpr>(expr->getIndex())->getValue() == "99");
  }
};
class ASTPostOrderVisitor : public ::testing::Test {
protected:
  std::shared_ptr<BlockStmt> blockStmt_;

  void SetUp() override {
    blockStmt_ = std::make_shared<BlockStmt>();

    // pi = {3.14, 3.14, 3.14};
    Type floatType(BuiltinTypeID::Float);
    std::shared_ptr<LiteralAccessExpr> pi =
        std::make_shared<LiteralAccessExpr>("3.14", BuiltinTypeID::Float);
    std::vector<std::shared_ptr<Expr>> initList = {pi, pi, pi};
    std::shared_ptr<VarDeclStmt> varDcl =
        std::make_shared<VarDeclStmt>(floatType, "u1", 3, "=", initList);

    blockStmt_->push_back(varDcl);

    // index to access pi[1]
    std::shared_ptr<LiteralAccessExpr> index0 =
        std::make_shared<LiteralAccessExpr>("0", BuiltinTypeID::Integer);
    std::shared_ptr<LiteralAccessExpr> index1 =
        std::make_shared<LiteralAccessExpr>("1", BuiltinTypeID::Integer);

    // f1 = f2+pi[1]
    std::shared_ptr<FieldAccessExpr> f1 = std::make_shared<FieldAccessExpr>("f1");
    std::shared_ptr<FieldAccessExpr> f2 = std::make_shared<FieldAccessExpr>("f2");
    std::shared_ptr<VarAccessExpr> varAccess = std::make_shared<VarAccessExpr>("pi", index1);
    std::shared_ptr<BinaryOperator> rightExpr =
        std::make_shared<BinaryOperator>(f2, "+", varAccess);
    std::shared_ptr<AssignmentExpr> assignExpr = std::make_shared<AssignmentExpr>(f1, rightExpr);
    std::shared_ptr<ExprStmt> exprStmt1 = std::make_shared<ExprStmt>(assignExpr);

    blockStmt_->push_back(exprStmt1);

    std::shared_ptr<VarAccessExpr> varAccess1 = std::make_shared<VarAccessExpr>("pi", index0);

    std::shared_ptr<LiteralAccessExpr> val0 =
        std::make_shared<LiteralAccessExpr>("0", BuiltinTypeID::Float);

    std::shared_ptr<BinaryOperator> equal =
        std::make_shared<BinaryOperator>(varAccess1, "==", val0);
    std::shared_ptr<ExprStmt> condStmt = std::make_shared<ExprStmt>(equal);
    std::shared_ptr<ReturnStmt> returnIf = std::make_shared<ReturnStmt>(f1);
    std::shared_ptr<ReturnStmt> returnElse = std::make_shared<ReturnStmt>(f2);

    std::shared_ptr<IfStmt> ifStmt = std::make_shared<IfStmt>(condStmt, returnIf, returnElse);
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

class ReplaceAssignVisitor : public ASTVisitorPostOrder, public NonCopyable {
protected:
public:
  ReplaceAssignVisitor() {}

  virtual ~ReplaceAssignVisitor() {}

  virtual std::shared_ptr<Expr>
  postVisitNode(std::shared_ptr<AssignmentExpr> const& expr) override {
    return std::make_shared<AssignmentExpr>(std::make_shared<FieldAccessExpr>("demo_out"),
                                            std::make_shared<FieldAccessExpr>("demo_in"));
  }
};

class CheckAssignVisitor : public ASTVisitorForwarding, public NonCopyable {

  bool result_ = true;

public:
  CheckAssignVisitor() {}

  bool result() const { return result_; }
  virtual ~CheckAssignVisitor() {}

  virtual void visit(std::shared_ptr<AssignmentExpr> const& expr) override {
    DAWN_ASSERT(isa<FieldAccessExpr>(*(expr->getLeft())));

    result_ = result_ && (std::dynamic_pointer_cast<FieldAccessExpr>(expr->getLeft())->getName() ==
                          "demo_out");
    result_ = result_ && (std::dynamic_pointer_cast<FieldAccessExpr>(expr->getRight())->getName() ==
                          "demo_in");
  }
};

TEST_F(ASTPostOrderVisitor, assignment) {

  ReplaceAssignVisitor repl;
  blockStmt_->acceptAndReplace(repl);

  CheckAssignVisitor checker;
  blockStmt_->accept(checker);

  ASSERT_TRUE(checker.result());
}

class ReplaceIfVisitor : public ASTVisitorPostOrder, public NonCopyable {
protected:
public:
  ReplaceIfVisitor() {}

  virtual ~ReplaceIfVisitor() {}

  virtual std::shared_ptr<Stmt> postVisitNode(std::shared_ptr<IfStmt> const& stmt) override {
    DAWN_ASSERT(isa<BinaryOperator>(*(stmt->getCondExpr())));

    std::shared_ptr<BinaryOperator> op =
        std::dynamic_pointer_cast<BinaryOperator>(stmt->getCondExpr());

    std::shared_ptr<ExprStmt> gt = std::make_shared<ExprStmt>(
        std::make_shared<BinaryOperator>(op->getLeft(), ">=", op->getRight()));

    std::shared_ptr<ExprStmt> ifStmt = std::make_shared<ExprStmt>(std::make_shared<NOPExpr>());
    std::shared_ptr<ExprStmt> elseStmt = std::make_shared<ExprStmt>(std::make_shared<NOPExpr>());

    return std::make_shared<IfStmt>(gt, ifStmt, elseStmt);
  }
};

class CheckIfVisitor : public ASTVisitorForwarding, public NonCopyable {

  bool result_ = true;

public:
  CheckIfVisitor() {}

  bool result() const { return result_; }
  virtual ~CheckIfVisitor() {}

  virtual void visit(std::shared_ptr<IfStmt> const& stmt) override {
    DAWN_ASSERT(isa<BinaryOperator>(*(stmt->getCondExpr())));

    std::shared_ptr<BinaryOperator> op =
        std::dynamic_pointer_cast<BinaryOperator>(stmt->getCondExpr());

    result_ = result_ && (std::string(op->getOp()) == ">=");

    DAWN_ASSERT(isa<ExprStmt>(*(stmt->getThenStmt())));
    std::shared_ptr<ExprStmt> thenExpr = std::dynamic_pointer_cast<ExprStmt>(stmt->getThenStmt());
    result_ = result_ && isa<NOPExpr>(*(thenExpr->getExpr()));
    std::shared_ptr<ExprStmt> elseExpr = std::dynamic_pointer_cast<ExprStmt>(stmt->getElseStmt());
    result_ = result_ && isa<NOPExpr>(*(elseExpr->getExpr()));
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
