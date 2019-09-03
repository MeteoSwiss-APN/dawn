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

#include "dawn/AST/AST.h"
#include "dawn/Support/StringSwitch.h"
#include "dawn/Support/Type.h"
#include "dawn/Support/Unreachable.h"
#include <functional>

namespace dawn {
namespace ast {

//===------------------------------------------------------------------------------------------===//
//     ExprReplacer
//===------------------------------------------------------------------------------------------===//

template <typename DataTraits>
ExprReplacer<DataTraits>::ExprReplacer(const std::shared_ptr<Expr<DataTraits>>& oldExpr,
                                       const std::shared_ptr<Expr<DataTraits>>& newExpr)
    : oldExpr_(oldExpr), newExpr_(newExpr) {}

template <typename DataTraits>
void ExprReplacer<DataTraits>::visit(const std::shared_ptr<BlockStmt<DataTraits>>& stmt) {
  for(const auto& s : stmt->getChildren())
    s->accept(*this);
}

template <typename DataTraits>
void ExprReplacer<DataTraits>::visit(const std::shared_ptr<ExprStmt<DataTraits>>& stmt) {
  if(stmt->getExpr() == oldExpr_)
    stmt->setExpr(newExpr_);
  else
    stmt->getExpr()->accept(*this);
}

template <typename DataTraits>
void ExprReplacer<DataTraits>::visit(const std::shared_ptr<ReturnStmt<DataTraits>>& stmt) {
  if(stmt->getExpr() == oldExpr_)
    stmt->setExpr(newExpr_);
  else
    stmt->getExpr()->accept(*this);
}

template <typename DataTraits>
void ExprReplacer<DataTraits>::visit(const std::shared_ptr<VarDeclStmt<DataTraits>>& stmt) {
  for(auto& expr : stmt->getInitList()) {
    if(expr == oldExpr_)
      expr = newExpr_;
    else
      expr->accept(*this);
  }
}

template <typename DataTraits>
void ExprReplacer<DataTraits>::visit(const std::shared_ptr<StencilCallDeclStmt<DataTraits>>& stmt) {
}

template <typename DataTraits>
void ExprReplacer<DataTraits>::visit(
    const std::shared_ptr<BoundaryConditionDeclStmt<DataTraits>>& stmt) {}

template <typename DataTraits>
void ExprReplacer<DataTraits>::visit(const std::shared_ptr<IfStmt<DataTraits>>& stmt) {
  for(const auto& s : stmt->getChildren())
    s->accept(*this);
}

template <typename DataTraits>
void ExprReplacer<DataTraits>::visit(const std::shared_ptr<UnaryOperator<DataTraits>>& expr) {
  if(expr->getOperand() == oldExpr_)
    expr->setOperand(newExpr_);
  else
    expr->getOperand()->accept(*this);
}

template <typename DataTraits>
void ExprReplacer<DataTraits>::visit(const std::shared_ptr<BinaryOperator<DataTraits>>& expr) {
  if(expr->getLeft() == oldExpr_)
    expr->setLeft(newExpr_);
  else
    expr->getLeft()->accept(*this);

  if(expr->getRight() == oldExpr_)
    expr->setRight(newExpr_);
  else
    expr->getRight()->accept(*this);
}

template <typename DataTraits>
void ExprReplacer<DataTraits>::visit(const std::shared_ptr<AssignmentExpr<DataTraits>>& expr) {
  if(expr->getLeft() == oldExpr_)
    expr->setLeft(newExpr_);
  else
    expr->getLeft()->accept(*this);

  if(expr->getRight() == oldExpr_)
    expr->setRight(newExpr_);
  else
    expr->getRight()->accept(*this);
}

template <typename DataTraits>
void ExprReplacer<DataTraits>::visit(const std::shared_ptr<TernaryOperator<DataTraits>>& expr) {
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

template <typename DataTraits>
void ExprReplacer<DataTraits>::visit(const std::shared_ptr<FunCallExpr<DataTraits>>& expr) {
  for(auto& e : expr->getArguments()) {
    if(e == oldExpr_)
      e = newExpr_;
    else
      e->accept(*this);
  }
}

template <typename DataTraits>
void ExprReplacer<DataTraits>::visit(const std::shared_ptr<StencilFunCallExpr<DataTraits>>& expr) {
  for(auto& e : expr->getArguments()) {
    if(e == oldExpr_)
      e = newExpr_;
    else
      e->accept(*this);
  }
}

template <typename DataTraits>
void ExprReplacer<DataTraits>::visit(const std::shared_ptr<StencilFunArgExpr<DataTraits>>& expr) {}

template <typename DataTraits>
void ExprReplacer<DataTraits>::visit(const std::shared_ptr<VarAccessExpr<DataTraits>>& expr) {
  if(expr->isArrayAccess()) {
    if(expr->getIndex() == oldExpr_)
      expr->setIndex(newExpr_);
    else
      expr->getIndex()->accept(*this);
  }
}

template <typename DataTraits>
void ExprReplacer<DataTraits>::visit(const std::shared_ptr<FieldAccessExpr<DataTraits>>& expr) {}

template <typename DataTraits>
void ExprReplacer<DataTraits>::visit(const std::shared_ptr<LiteralAccessExpr<DataTraits>>& expr) {}

template <typename DataTraits>
void replaceOldExprWithNewExprInStmt(const std::shared_ptr<Stmt<DataTraits>>& stmt,
                                     const std::shared_ptr<Expr<DataTraits>>& oldExpr,
                                     const std::shared_ptr<Expr<DataTraits>>& newExpr) {
  ExprReplacer<DataTraits> replacer(oldExpr, newExpr);
  stmt->accept(replacer);
}

//===------------------------------------------------------------------------------------------===//
//     StmtReplacer
//===------------------------------------------------------------------------------------------===//

template <typename DataTraits>
StmtReplacer<DataTraits>::StmtReplacer(const std::shared_ptr<Stmt<DataTraits>>& oldStmt,
                                       const std::shared_ptr<Stmt<DataTraits>>& newStmt)
    : oldStmt_(oldStmt), newStmt_(newStmt) {}

template <typename DataTraits>
void StmtReplacer<DataTraits>::visit(const std::shared_ptr<BlockStmt<DataTraits>>& stmt) {
  for(auto& s : stmt->getStatements()) {
    if(s == oldStmt_)
      s = newStmt_;
    else
      s->accept(*this);
  }
}

template <typename DataTraits>
void StmtReplacer<DataTraits>::visit(const std::shared_ptr<IfStmt<DataTraits>>& stmt) {
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

template <typename DataTraits>
void replaceOldStmtWithNewStmtInStmt(const std::shared_ptr<Stmt<DataTraits>>& stmt,
                                     const std::shared_ptr<Stmt<DataTraits>>& oldStmt,
                                     const std::shared_ptr<Stmt<DataTraits>>& newStmt) {
  StmtReplacer<DataTraits> replacer(oldStmt, newStmt);
  stmt->accept(replacer);
}

//===------------------------------------------------------------------------------------------===//
//     ExprEvaluator
//===------------------------------------------------------------------------------------------===//

namespace {

enum OpKind {
  OK_Plus,         // std::plus
  OK_Minus,        // std::minus
  OK_Multiplies,   // std::multiplies
  OK_Divides,      // std::divides
  OK_Modulus,      // std::modulus
  OK_Negate,       // std::negate
  OK_EqualTo,      // std::equal_to
  OK_NotEqualTo,   // std::not_equal_to
  OK_Greater,      // std::greater
  OK_Less,         // std::less
  OK_GreaterEqual, // std::greater_equal
  OK_LessEqual,    // std::less_equal
  OK_LogicalAnd,   // std::logical_and
  OK_LogicalOr,    // std::logical_or
  OK_LogicalNot,   // std::logical_not

  Op_Nop
};

/// @brief String to OpKind
///
/// Note that we do not evaluate bit-wise operations and the modulo operator
OpKind toOpKind(const char* op) {
  return StringSwitch<OpKind>(op)
      .Case("+", OK_Plus)
      .Case("-", OK_Minus)
      .Case("*", OK_Multiplies)
      .Case("/", OK_Divides)
      .Case("-", OK_Negate)
      .Case("==", OK_EqualTo)
      .Case("!=", OK_NotEqualTo)
      .Case(">", OK_Greater)
      .Case("<", OK_Less)
      .Case(">=", OK_GreaterEqual)
      .Case("<=", OK_LessEqual)
      .Case("&&", OK_LogicalAnd)
      .Case("||", OK_LogicalOr)
      .Case("!", OK_LogicalNot)
      .Default(Op_Nop);
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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
bool evalUnary(const char* opStr, double& result, double operand) {
  switch(toOpKind(opStr)) {
  case OK_Negate:
  case OK_Minus:
    return evalImpl<std::negate<double>>(result, operand);
  case OK_LogicalNot:
    return evalImpl<std::logical_not<double>>(result, operand);
  default:
    return false;
  }
}
#pragma GCC diagnostic push

//
// Binary operations
//
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
bool evalBinary(const char* opStr, double& result, double op1, double op2) {
  switch(toOpKind(opStr)) {
  case OK_Plus:
    return evalImpl<std::plus<double>>(result, op1, op2);
  case OK_Minus:
    return evalImpl<std::minus<double>>(result, op1, op2);
  case OK_Multiplies:
    return evalImpl<std::multiplies<double>>(result, op1, op2);
  case OK_Divides:
    return evalImpl<std::divides<double>>(result, op1, op2);
  case OK_EqualTo:
    return evalImpl<std::equal_to<double>>(result, op1, op2);
  case OK_NotEqualTo:
    return evalImpl<std::not_equal_to<double>>(result, op1, op2);
  case OK_Greater:
    return evalImpl<std::greater<double>>(result, op1, op2);
  case OK_Less:
    return evalImpl<std::less<double>>(result, op1, op2);
  case OK_GreaterEqual:
    return evalImpl<std::greater_equal<double>>(result, op1, op2);
  case OK_LessEqual:
    return evalImpl<std::less_equal<double>>(result, op1, op2);
  case OK_LogicalAnd:
    return evalImpl<std::logical_and<double>>(result, op1, op2);
  case OK_LogicalOr:
    return evalImpl<std::logical_or<double>>(result, op1, op2);
  default:
    return false;
  }
}
#pragma GCC diagnostic push

/// @brief Evaluate `expr` treating everything as `double`s
///
/// Given that we only have 32-bit integers it should be safe to treat them as double.
template <typename DataTraits>
class ExprEvaluator : public ASTVisitor<DataTraits> {
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

  void visit(const std::shared_ptr<BlockStmt<DataTraits>>& stmt) override {
    dawn_unreachable("cannot evaluate stmt");
  }

  void visit(const std::shared_ptr<ExprStmt<DataTraits>>& stmt) override {
    dawn_unreachable("cannot evaluate stmt");
  }

  void visit(const std::shared_ptr<ReturnStmt<DataTraits>>& stmt) override {
    dawn_unreachable("cannot evaluate stmt");
  }

  void visit(const std::shared_ptr<VarDeclStmt<DataTraits>>& stmt) override {
    dawn_unreachable("cannot evaluate stmt");
  }

  void visit(const std::shared_ptr<StencilCallDeclStmt<DataTraits>>& stmt) override {
    dawn_unreachable("cannot evaluate stmt");
  }

  void visit(const std::shared_ptr<BoundaryConditionDeclStmt<DataTraits>>& stmt) override {
    dawn_unreachable("cannot evaluate stmt");
  }

  void visit(const std::shared_ptr<IfStmt<DataTraits>>& stmt) override {
    dawn_unreachable("cannot evaluate stmt");
  }

  void visit(const std::shared_ptr<UnaryOperator<DataTraits>>& expr) override {
    if(!valid_)
      return;

    ExprEvaluator evaluator(variableMap_);
    expr->getOperand()->accept(evaluator);

    if((valid_ = evaluator.isValid()))
      valid_ = evalUnary(expr->getOp(), result_, evaluator.getResult());
  }

  void visit(const std::shared_ptr<BinaryOperator<DataTraits>>& expr) override {
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

  void visit(const std::shared_ptr<AssignmentExpr<DataTraits>>& expr) override {
    // In C++ this can actually be evaluated but we don't do it for now
    valid_ = false;
  }

  void visit(const std::shared_ptr<TernaryOperator<DataTraits>>& expr) override {
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

  void visit(const std::shared_ptr<FunCallExpr<DataTraits>>& expr) override { valid_ = false; }

  void visit(const std::shared_ptr<StencilFunCallExpr<DataTraits>>& expr) override {
    valid_ = false;
  }

  void visit(const std::shared_ptr<StencilFunArgExpr<DataTraits>>& expr) override {
    valid_ = false;
  }

  void visit(const std::shared_ptr<VarAccessExpr<DataTraits>>& expr) override {
    auto it = variableMap_.find(expr->getName());
    if(it != variableMap_.end())
      result_ = it->second;
    else
      valid_ = false;
  }

  void visit(const std::shared_ptr<FieldAccessExpr<DataTraits>>& expr) override { valid_ = false; }

  void visit(const std::shared_ptr<LiteralAccessExpr<DataTraits>>& expr) override {
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

template <typename DataTraits, class T>
bool evalExprImpl(const std::shared_ptr<Expr<DataTraits>>& expr, T& result,
                  const std::unordered_map<std::string, double>& variableMap) {
  ExprEvaluator<DataTraits> evaluator(variableMap);
  expr->accept(evaluator);
  if(evaluator.isValid()) {
    result = static_cast<T>(evaluator.getResult());
    return true;
  }
  return false;
}

} // anonymous namespace

template <typename DataTraits>
bool evalExprAsDouble(const std::shared_ptr<Expr<DataTraits>>& expr, double& result,
                      const std::unordered_map<std::string, double>& variableMap) {
  return evalExprImpl(expr, result, variableMap);
}

template <typename DataTraits>
bool evalExprAsInteger(const std::shared_ptr<Expr<DataTraits>>& expr, int& result,
                       const std::unordered_map<std::string, double>& variableMap) {
  return evalExprImpl(expr, result, variableMap);
}

template <typename DataTraits>
bool evalExprAsBoolean(const std::shared_ptr<Expr<DataTraits>>& expr, bool& result,
                       const std::unordered_map<std::string, double>& variableMap) {
  return evalExprImpl(expr, result, variableMap);
}

} // namespace ast
} // namespace dawn
