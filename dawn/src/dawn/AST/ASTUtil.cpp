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

#include "dawn/AST/ASTUtil.h"
#include "dawn/AST/AST.h"
#include "dawn/AST/ASTVisitor.h"
#include "dawn/Support/StringSwitch.h"
#include "dawn/Support/Unreachable.h"
#include <functional>

namespace dawn {
namespace ast {
//===------------------------------------------------------------------------------------------===//
//     ExprReplacer
//===------------------------------------------------------------------------------------------===//

namespace {

/// @brief Replace `oldExpr` with `newExpr`
class ExprReplacer : public ASTVisitor {
  std::shared_ptr<Expr> oldExpr_;
  std::shared_ptr<Expr> newExpr_;

public:
  ExprReplacer(const std::shared_ptr<Expr>& oldExpr, const std::shared_ptr<Expr>& newExpr)
      : oldExpr_(oldExpr), newExpr_(newExpr) {}

  void visit(const std::shared_ptr<BlockStmt>& stmt) override {
    for(const auto& s : stmt->getChildren())
      s->accept(*this);
  }

  void visit(const std::shared_ptr<ExprStmt>& stmt) override {
    if(stmt->getExpr() == oldExpr_)
      stmt->setExpr(newExpr_);
    else
      stmt->getExpr()->accept(*this);
  }

  void visit(const std::shared_ptr<ReturnStmt>& stmt) override {
    if(stmt->getExpr() == oldExpr_)
      stmt->setExpr(newExpr_);
    else
      stmt->getExpr()->accept(*this);
  }

  void visit(const std::shared_ptr<VarDeclStmt>& stmt) override {
    for(auto& expr : stmt->getInitList()) {
      if(expr == oldExpr_)
        expr = newExpr_;
      else
        expr->accept(*this);
    }
  }

  void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) override {}
  void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) override {}
  void visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) override {}
  void visit(const std::shared_ptr<ReductionOverNeighborExpr>& expr) override {
    for(const auto& s : expr->getChildren())
      s->accept(*this);
  }

  void visit(const std::shared_ptr<IfStmt>& stmt) override {
    for(const auto& s : stmt->getChildren())
      s->accept(*this);
  }

  void visit(const std::shared_ptr<UnaryOperator>& expr) override {
    if(expr->getOperand() == oldExpr_)
      expr->setOperand(newExpr_);
    else
      expr->getOperand()->accept(*this);
  }

  void visit(const std::shared_ptr<BinaryOperator>& expr) override {
    if(expr->getLeft() == oldExpr_)
      expr->setLeft(newExpr_);
    else
      expr->getLeft()->accept(*this);

    if(expr->getRight() == oldExpr_)
      expr->setRight(newExpr_);
    else
      expr->getRight()->accept(*this);
  }

  void visit(const std::shared_ptr<AssignmentExpr>& expr) override {
    if(expr->getLeft() == oldExpr_)
      expr->setLeft(newExpr_);
    else
      expr->getLeft()->accept(*this);

    if(expr->getRight() == oldExpr_)
      expr->setRight(newExpr_);
    else
      expr->getRight()->accept(*this);
  }

  void visit(const std::shared_ptr<TernaryOperator>& expr) override {
    if(expr->getCondition() == oldExpr_)
      expr->setCondition(newExpr_);
    else
      expr->getCondition()->accept(*this);

    if(expr->getLeft() == oldExpr_)
      expr->setLeft(newExpr_);
    else
      expr->getLeft()->accept(*this);

    if(expr->getRight() == oldExpr_)
      expr->setRight(newExpr_);
    else
      expr->getRight()->accept(*this);
  }

  void visit(const std::shared_ptr<FunCallExpr>& expr) override {
    for(auto& e : expr->getArguments()) {
      if(e == oldExpr_)
        e = newExpr_;
      else
        e->accept(*this);
    }
  }

  void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override {
    for(auto& e : expr->getArguments()) {
      if(e == oldExpr_)
        e = newExpr_;
      else
        e->accept(*this);
    }
  }

  void visit(const std::shared_ptr<StencilFunArgExpr>& expr) override {}

  void visit(const std::shared_ptr<VarAccessExpr>& expr) override {
    if(expr->isArrayAccess()) {
      if(expr->getIndex() == oldExpr_)
        expr->setIndex(newExpr_);
      else
        expr->getIndex()->accept(*this);
    }
  }

  void visit(const std::shared_ptr<FieldAccessExpr>& expr) override {}
  void visit(const std::shared_ptr<LiteralAccessExpr>& expr) override {}
};

} // anonymous namespace

void replaceOldExprWithNewExprInStmt(const std::shared_ptr<Stmt>& stmt,
                                     const std::shared_ptr<Expr>& oldExpr,
                                     const std::shared_ptr<Expr>& newExpr) {
  ExprReplacer replacer(oldExpr, newExpr);
  stmt->accept(replacer);
}

//===------------------------------------------------------------------------------------------===//
//     StmtReplacer
//===------------------------------------------------------------------------------------------===//

namespace {

/// @brief Replace `oldStmt` with `newStmt`
class StmtReplacer : public ASTVisitorForwarding {
  std::shared_ptr<Stmt> oldStmt_;
  std::shared_ptr<Stmt> newStmt_;

public:
  StmtReplacer(const std::shared_ptr<Stmt>& oldStmt, const std::shared_ptr<Stmt>& newStmt)
      : oldStmt_(oldStmt), newStmt_(newStmt) {}

  void visit(const std::shared_ptr<BlockStmt>& stmt) override {
    for(auto it = stmt->getStatements().cbegin(); it != stmt->getStatements().cend(); ++it) {
      if(*it == oldStmt_) {
        stmt->substitute(it, std::move(newStmt_));
      } else
        (*it)->accept(*this);
    }
  }

  void visit(const std::shared_ptr<IfStmt>& stmt) override {
    if(stmt->getThenStmt() == oldStmt_)
      stmt->setThenStmt(newStmt_);
    else
      stmt->getThenStmt()->accept(*this);

    if(stmt->hasElse()) {
      if(stmt->getElseStmt() == oldStmt_)
        stmt->setElseStmt(newStmt_);
      else
        stmt->getElseStmt()->accept(*this);
    }
  }
};

} // anonymous namespace

void replaceOldStmtWithNewStmtInStmt(const std::shared_ptr<Stmt>& stmt,
                                     const std::shared_ptr<Stmt>& oldStmt,
                                     const std::shared_ptr<Stmt>& newStmt) {
  StmtReplacer replacer(oldStmt, newStmt);
  stmt->accept(replacer);
}

//===------------------------------------------------------------------------------------------===//
//     ExprEvaluator
//===------------------------------------------------------------------------------------------===//

namespace {

enum class OpKind {
  Plus,         // std::plus
  Minus,        // std::minus
  Multiplies,   // std::multiplies
  Divides,      // std::divides
  Modulus,      // std::modulus
  Negate,       // std::negate
  EqualTo,      // std::equal_to
  NotEqualTo,   // std::not_equal_to
  Greater,      // std::greater
  Less,         // std::less
  GreaterEqual, // std::greater_equal
  LessEqual,    // std::less_equal
  LogicalAnd,   // std::logical_and
  LogicalOr,    // std::logical_or
  LogicalNot,   // std::logical_not

  Nop
};

/// @brief String to OpKind
///
/// Note that we do not evaluate bit-wise operations and the modulo operator
OpKind toOpKind(const char* op) {
  return StringSwitch<OpKind>(op)
      .Case("+", OpKind::Plus)
      .Case("-", OpKind::Minus)
      .Case("*", OpKind::Multiplies)
      .Case("/", OpKind::Divides)
      .Case("-", OpKind::Negate)
      .Case("==", OpKind::EqualTo)
      .Case("!=", OpKind::NotEqualTo)
      .Case(">", OpKind::Greater)
      .Case("<", OpKind::Less)
      .Case(">=", OpKind::GreaterEqual)
      .Case("<=", OpKind::LessEqual)
      .Case("&&", OpKind::LogicalAnd)
      .Case("||", OpKind::LogicalOr)
      .Case("!", OpKind::LogicalNot)
      .Default(OpKind::Nop);
}

/// @brief Evaluate the given functor
template <class Functor, class ResultType, class... ValueTypes>
static bool evalImpl(ResultType& result, ValueTypes... operands) {
  result = Functor()(operands...);
  return true;
}

//
// Unary operations
//
bool evalUnary(const char* opStr, double& result, double operand) {
  switch(toOpKind(opStr)) {
  case OpKind::Negate:
  case OpKind::Minus:
    return evalImpl<std::negate<double>>(result, operand);
  case OpKind::LogicalNot:
    return evalImpl<std::logical_not<double>>(result, operand);
  default:
    return false;
  }
}

//
// Binary operations
//
bool evalBinary(const char* opStr, double& result, double op1, double op2) {
  switch(toOpKind(opStr)) {
  case OpKind::Plus:
    return evalImpl<std::plus<double>>(result, op1, op2);
  case OpKind::Minus:
    return evalImpl<std::minus<double>>(result, op1, op2);
  case OpKind::Multiplies:
    return evalImpl<std::multiplies<double>>(result, op1, op2);
  case OpKind::Divides:
    return evalImpl<std::divides<double>>(result, op1, op2);
  case OpKind::EqualTo:
    return evalImpl<std::equal_to<double>>(result, op1, op2);
  case OpKind::NotEqualTo:
    return evalImpl<std::not_equal_to<double>>(result, op1, op2);
  case OpKind::Greater:
    return evalImpl<std::greater<double>>(result, op1, op2);
  case OpKind::Less:
    return evalImpl<std::less<double>>(result, op1, op2);
  case OpKind::GreaterEqual:
    return evalImpl<std::greater_equal<double>>(result, op1, op2);
  case OpKind::LessEqual:
    return evalImpl<std::less_equal<double>>(result, op1, op2);
  case OpKind::LogicalAnd:
    return evalImpl<std::logical_and<double>>(result, op1, op2);
  case OpKind::LogicalOr:
    return evalImpl<std::logical_or<double>>(result, op1, op2);
  default:
    return false;
  }
}

/// @brief Evaluate `expr` treating everything as `double`s
///
/// Given that we only have 32-bit integers it should be safe to treat them as double.
class ExprEvaluator : public ASTVisitor {
  const std::unordered_map<std::string, double>& variableMap_;

  bool valid_;
  double result_;

public:
  ExprEvaluator(const std::unordered_map<std::string, double>& variableMap)
      : variableMap_(variableMap), valid_(true), result_(double()) {}

  /// @brief Get the result of the evaluation
  double getResult() const {
    DAWN_ASSERT(valid_);
    return result_;
  }

  /// @brief Check if evaluation succeeded
  bool isValid() const { return valid_; }

  void visit(const std::shared_ptr<BlockStmt>& stmt) override {
    dawn_unreachable("cannot evaluate stmt");
  }

  void visit(const std::shared_ptr<ExprStmt>& stmt) override {
    dawn_unreachable("cannot evaluate stmt");
  }

  void visit(const std::shared_ptr<ReturnStmt>& stmt) override {
    dawn_unreachable("cannot evaluate stmt");
  }

  void visit(const std::shared_ptr<VarDeclStmt>& stmt) override {
    dawn_unreachable("cannot evaluate stmt");
  }

  void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) override {
    dawn_unreachable("cannot evaluate stmt");
  }

  void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) override {
    dawn_unreachable("cannot evaluate stmt");
  }

  void visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) override {
    dawn_unreachable("cannot evaluate stmt");
  }
  void visit(const std::shared_ptr<ReductionOverNeighborExpr>& expr) override {
    dawn_unreachable("cannot evaluate expr");
  }

  void visit(const std::shared_ptr<IfStmt>& stmt) override {
    dawn_unreachable("cannot evaluate stmt");
  }

  void visit(const std::shared_ptr<UnaryOperator>& expr) override {
    if(!valid_)
      return;

    ExprEvaluator evaluator(variableMap_);
    expr->getOperand()->accept(evaluator);

    if((valid_ = evaluator.isValid()))
      valid_ = evalUnary(expr->getOp(), result_, evaluator.getResult());
  }

  void visit(const std::shared_ptr<BinaryOperator>& expr) override {
    if(!valid_)
      return;

    ExprEvaluator evaluatorLeft(variableMap_);
    expr->getLeft()->accept(evaluatorLeft);

    ExprEvaluator evaluatorRight(variableMap_);
    expr->getRight()->accept(evaluatorRight);

    if((valid_ = evaluatorLeft.isValid() && evaluatorRight.isValid()))
      valid_ =
          evalBinary(expr->getOp(), result_, evaluatorLeft.getResult(), evaluatorRight.getResult());
  }

  void visit(const std::shared_ptr<AssignmentExpr>& expr) override {
    // In C++ this can actually be evaluated but we don't do it for now
    valid_ = false;
  }

  void visit(const std::shared_ptr<TernaryOperator>& expr) override {
    if(!valid_)
      return;

    ExprEvaluator evaluatorCond(variableMap_);
    expr->getCondition()->accept(evaluatorCond);

    if(!evaluatorCond.isValid())
      return;

    if(evaluatorCond.getResult()) {
      ExprEvaluator evaluator(variableMap_);
      expr->getLeft()->accept(evaluator);
      if((valid_ = evaluator.isValid()))
        result_ = evaluator.getResult();
    } else {
      ExprEvaluator evaluator(variableMap_);
      expr->getRight()->accept(evaluator);
      if((valid_ = evaluator.isValid()))
        result_ = evaluator.getResult();
    }
  }

  void visit(const std::shared_ptr<FunCallExpr>& expr) override { valid_ = false; }

  void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override { valid_ = false; }

  void visit(const std::shared_ptr<StencilFunArgExpr>& expr) override { valid_ = false; }

  void visit(const std::shared_ptr<VarAccessExpr>& expr) override {
    auto it = variableMap_.find(expr->getName());
    if(it != variableMap_.end())
      result_ = it->second;
    else
      valid_ = false;
  }

  void visit(const std::shared_ptr<FieldAccessExpr>& expr) override { valid_ = false; }

  void visit(const std::shared_ptr<LiteralAccessExpr>& expr) override {
    switch(expr->getBuiltinType()) {
    case BuiltinTypeID::Boolean:
      result_ = expr->getValue() == "1" || expr->getValue() == "true";
      break;
    case BuiltinTypeID::Integer:
      result_ = std::atoi(expr->getValue().c_str());
      break;
    case BuiltinTypeID::Float:
      result_ = std::atof(expr->getValue().c_str());
      break;
    default:
      valid_ = false;
    }
  }
};

template <class T>
bool evalExprImpl(const std::shared_ptr<Expr>& expr, T& result,
                  const std::unordered_map<std::string, double>& variableMap) {
  ExprEvaluator evaluator(variableMap);
  expr->accept(evaluator);
  if(evaluator.isValid()) {
    result = static_cast<T>(evaluator.getResult());
    return true;
  }
  return false;
}

} // anonymous namespace

bool evalExprAsDouble(const std::shared_ptr<Expr>& expr, double& result,
                      const std::unordered_map<std::string, double>& variableMap) {
  return evalExprImpl(expr, result, variableMap);
}

bool evalExprAsInteger(const std::shared_ptr<Expr>& expr, int& result,
                       const std::unordered_map<std::string, double>& variableMap) {
  return evalExprImpl(expr, result, variableMap);
}

bool evalExprAsBoolean(const std::shared_ptr<Expr>& expr, bool& result,
                       const std::unordered_map<std::string, double>& variableMap) {
  return evalExprImpl(expr, result, variableMap);
}

} // namespace ast
} // namespace dawn
