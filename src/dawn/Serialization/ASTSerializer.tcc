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

#include "dawn/SIR/ASTStmt.h"
#include <google/protobuf/util/json_util.h>
#include <list>
#include <memory>

using namespace dawn;
using namespace ast;

template <typename DataTraits>
ProtoStmtBuilder<DataTraits>::ProtoStmtBuilder(dawn::proto::statements::Stmt* stmtProto) {
  currentStmtProto_.push(stmtProto);
}

template <typename DataTraits>
ProtoStmtBuilder<DataTraits>::ProtoStmtBuilder(dawn::proto::statements::Expr* exprProto) {
  currentExprProto_.push(exprProto);
}

template <typename DataTraits>
dawn::proto::statements::Stmt* ProtoStmtBuilder<DataTraits>::getCurrentStmtProto() {
  DAWN_ASSERT(!currentStmtProto_.empty());
  return currentStmtProto_.top();
}

template <typename DataTraits>
dawn::proto::statements::Expr* ProtoStmtBuilder<DataTraits>::getCurrentExprProto() {
  DAWN_ASSERT(!currentExprProto_.empty());
  return currentExprProto_.top();
}

template <typename DataTraits>
void ProtoStmtBuilder<DataTraits>::visit(const std::shared_ptr<BlockStmt<DataTraits>>& stmt) {
  auto protoStmt = getCurrentStmtProto()->mutable_block_stmt();

  for(const auto& s : stmt->getStatements()) {
    currentStmtProto_.push(protoStmt->add_statements());
    s->accept(*this);
    currentStmtProto_.pop();
  }

  setLocation(protoStmt->mutable_loc(), stmt->getSourceLocation());
  protoStmt->set_id(stmt->getID());
}

template <typename DataTraits>
void ProtoStmtBuilder<DataTraits>::visit(const std::shared_ptr<ExprStmt<DataTraits>>& stmt) {
  auto protoStmt = getCurrentStmtProto()->mutable_expr_stmt();
  currentExprProto_.push(protoStmt->mutable_expr());
  stmt->getExpr()->accept(*this);
  currentExprProto_.pop();

  setLocation(protoStmt->mutable_loc(), stmt->getSourceLocation());
  protoStmt->set_id(stmt->getID());
}

template <typename DataTraits>
void ProtoStmtBuilder<DataTraits>::visit(const std::shared_ptr<ReturnStmt<DataTraits>>& stmt) {
  auto protoStmt = getCurrentStmtProto()->mutable_return_stmt();

  currentExprProto_.push(protoStmt->mutable_expr());
  stmt->getExpr()->accept(*this);
  currentExprProto_.pop();

  setLocation(protoStmt->mutable_loc(), stmt->getSourceLocation());
  protoStmt->set_id(stmt->getID());
}

template <typename DataTraits>
void ProtoStmtBuilder<DataTraits>::visit(const std::shared_ptr<VarDeclStmt<DataTraits>>& stmt) {
  auto protoStmt = getCurrentStmtProto()->mutable_var_decl_stmt();

  if(stmt->getType().isBuiltinType())
    setBuiltinType(protoStmt->mutable_type()->mutable_builtin_type(),
                   stmt->getType().getBuiltinTypeID());
  else
    protoStmt->mutable_type()->set_name(stmt->getType().getName());
  protoStmt->mutable_type()->set_is_const(stmt->getType().isConst());
  protoStmt->mutable_type()->set_is_volatile(stmt->getType().isVolatile());

  protoStmt->set_name(stmt->getName());
  protoStmt->set_dimension(stmt->getDimension());
  protoStmt->set_op(stmt->getOp());

  for(const auto& expr : stmt->getInitList()) {
    currentExprProto_.push(protoStmt->add_init_list());
    expr->accept(*this);
    currentExprProto_.pop();
  }

  setLocation(protoStmt->mutable_loc(), stmt->getSourceLocation());
  protoStmt->set_id(stmt->getID());
}

template <typename DataTraits>
void ProtoStmtBuilder<DataTraits>::visit(
    const std::shared_ptr<StencilCallDeclStmt<DataTraits>>& stmt) {
  auto protoStmt = getCurrentStmtProto()->mutable_stencil_call_decl_stmt();

  dawn::ast::StencilCall* stencilCall = stmt->getStencilCall().get();
  dawn::proto::statements::StencilCall* stencilCallProto = protoStmt->mutable_stencil_call();

  // StencilCall.Loc
  setLocation(stencilCallProto->mutable_loc(), stencilCall->Loc);

  // StencilCall.Callee
  stencilCallProto->set_callee(stencilCall->Callee);

  // StencilCall.Args
  for(const auto& argName : stencilCall->Args) {
    stencilCallProto->add_arguments(argName);
  }

  setLocation(protoStmt->mutable_loc(), stmt->getSourceLocation());
  protoStmt->set_id(stmt->getID());
}

template <typename DataTraits>
void ProtoStmtBuilder<DataTraits>::visit(
    const std::shared_ptr<BoundaryConditionDeclStmt<DataTraits>>& stmt) {
  auto protoStmt = getCurrentStmtProto()->mutable_boundary_condition_decl_stmt();
  protoStmt->set_functor(stmt->getFunctor());

  for(const auto& fieldName : stmt->getFields())
    protoStmt->add_fields(fieldName);

  setLocation(protoStmt->mutable_loc(), stmt->getSourceLocation());
  protoStmt->set_id(stmt->getID());
}

template <typename DataTraits>
void ProtoStmtBuilder<DataTraits>::visit(const std::shared_ptr<IfStmt<DataTraits>>& stmt) {
  auto protoStmt = getCurrentStmtProto()->mutable_if_stmt();

  currentStmtProto_.push(protoStmt->mutable_cond_part());
  stmt->getCondStmt()->accept(*this);
  currentStmtProto_.pop();

  currentStmtProto_.push(protoStmt->mutable_then_part());
  stmt->getThenStmt()->accept(*this);
  currentStmtProto_.pop();

  currentStmtProto_.push(protoStmt->mutable_else_part());
  if(stmt->getElseStmt())
    stmt->getElseStmt()->accept(*this);
  currentStmtProto_.pop();

  setLocation(protoStmt->mutable_loc(), stmt->getSourceLocation());
  protoStmt->set_id(stmt->getID());
}

template <typename DataTraits>
void ProtoStmtBuilder<DataTraits>::visit(const std::shared_ptr<UnaryOperator<DataTraits>>& expr) {
  auto protoExpr = getCurrentExprProto()->mutable_unary_operator();
  protoExpr->set_op(expr->getOp());

  currentExprProto_.push(protoExpr->mutable_operand());
  expr->getOperand()->accept(*this);
  currentExprProto_.pop();

  setLocation(protoExpr->mutable_loc(), expr->getSourceLocation());
  protoExpr->set_id(expr->getID());
}

template <typename DataTraits>
void ProtoStmtBuilder<DataTraits>::visit(const std::shared_ptr<BinaryOperator<DataTraits>>& expr) {
  auto protoExpr = getCurrentExprProto()->mutable_binary_operator();
  protoExpr->set_op(expr->getOp());

  currentExprProto_.push(protoExpr->mutable_left());
  expr->getLeft()->accept(*this);
  currentExprProto_.pop();

  currentExprProto_.push(protoExpr->mutable_right());
  expr->getRight()->accept(*this);
  currentExprProto_.pop();

  setLocation(protoExpr->mutable_loc(), expr->getSourceLocation());
  protoExpr->set_id(expr->getID());
}

template <typename DataTraits>
void ProtoStmtBuilder<DataTraits>::visit(const std::shared_ptr<AssignmentExpr<DataTraits>>& expr) {
  auto protoExpr = getCurrentExprProto()->mutable_assignment_expr();
  protoExpr->set_op(expr->getOp());

  currentExprProto_.push(protoExpr->mutable_left());
  expr->getLeft()->accept(*this);
  currentExprProto_.pop();

  currentExprProto_.push(protoExpr->mutable_right());
  expr->getRight()->accept(*this);
  currentExprProto_.pop();

  setLocation(protoExpr->mutable_loc(), expr->getSourceLocation());
  protoExpr->set_id(expr->getID());
}

template <typename DataTraits>
void ProtoStmtBuilder<DataTraits>::visit(const std::shared_ptr<TernaryOperator<DataTraits>>& expr) {
  auto protoExpr = getCurrentExprProto()->mutable_ternary_operator();

  currentExprProto_.push(protoExpr->mutable_cond());
  expr->getCondition()->accept(*this);
  currentExprProto_.pop();

  currentExprProto_.push(protoExpr->mutable_left());
  expr->getLeft()->accept(*this);
  currentExprProto_.pop();

  currentExprProto_.push(protoExpr->mutable_right());
  expr->getRight()->accept(*this);
  currentExprProto_.pop();

  setLocation(protoExpr->mutable_loc(), expr->getSourceLocation());
  protoExpr->set_id(expr->getID());
}

template <typename DataTraits>
void ProtoStmtBuilder<DataTraits>::visit(const std::shared_ptr<FunCallExpr<DataTraits>>& expr) {
  auto protoExpr = getCurrentExprProto()->mutable_fun_call_expr();
  protoExpr->set_callee(expr->getCallee());

  for(const auto& arg : expr->getArguments()) {
    currentExprProto_.push(protoExpr->add_arguments());
    arg->accept(*this);
    currentExprProto_.pop();
  }

  setLocation(protoExpr->mutable_loc(), expr->getSourceLocation());
  protoExpr->set_id(expr->getID());
}

template <typename DataTraits>
void ProtoStmtBuilder<DataTraits>::visit(
    const std::shared_ptr<StencilFunCallExpr<DataTraits>>& expr) {
  auto protoExpr = getCurrentExprProto()->mutable_stencil_fun_call_expr();
  protoExpr->set_callee(expr->getCallee());

  for(const auto& arg : expr->getArguments()) {
    currentExprProto_.push(protoExpr->add_arguments());
    arg->accept(*this);
    currentExprProto_.pop();
  }

  setLocation(protoExpr->mutable_loc(), expr->getSourceLocation());
  protoExpr->set_id(expr->getID());
}

template <typename DataTraits>
void ProtoStmtBuilder<DataTraits>::visit(
    const std::shared_ptr<StencilFunArgExpr<DataTraits>>& expr) {
  auto protoExpr = getCurrentExprProto()->mutable_stencil_fun_arg_expr();

  protoExpr->mutable_dimension()->set_direction(
      expr->getDimension() == -1
          ? dawn::proto::statements::Dimension::Invalid
          : static_cast<dawn::proto::statements::Dimension_Direction>(expr->getDimension()));
  protoExpr->set_offset(expr->getOffset());
  protoExpr->set_argument_index(expr->getArgumentIndex());

  setLocation(protoExpr->mutable_loc(), expr->getSourceLocation());
  protoExpr->set_id(expr->getID());
}

template <typename DataTraits>
void ProtoStmtBuilder<DataTraits>::visit(const std::shared_ptr<VarAccessExpr<DataTraits>>& expr) {
  auto protoExpr = getCurrentExprProto()->mutable_var_access_expr();

  protoExpr->set_name(expr->getName());
  protoExpr->set_is_external(expr->isExternal());

  if(expr->isArrayAccess()) {
    currentExprProto_.push(protoExpr->mutable_index());
    expr->getIndex()->accept(*this);
    currentExprProto_.pop();
  }

  setLocation(protoExpr->mutable_loc(), expr->getSourceLocation());
  protoExpr->set_id(expr->getID());
}

template <typename DataTraits>
void ProtoStmtBuilder<DataTraits>::visit(const std::shared_ptr<FieldAccessExpr<DataTraits>>& expr) {
  auto protoExpr = getCurrentExprProto()->mutable_field_access_expr();

  protoExpr->set_name(expr->getName());

  for(int offset : expr->getOffset())
    protoExpr->add_offset(offset);

  for(int argOffset : expr->getArgumentOffset())
    protoExpr->add_argument_offset(argOffset);

  for(int argMap : expr->getArgumentMap())
    protoExpr->add_argument_map(argMap);

  protoExpr->set_negate_offset(expr->negateOffset());

  setLocation(protoExpr->mutable_loc(), expr->getSourceLocation());
  protoExpr->set_id(expr->getID());
}

template <typename DataTraits>
void ProtoStmtBuilder<DataTraits>::visit(
    const std::shared_ptr<LiteralAccessExpr<DataTraits>>& expr) {
  auto protoExpr = getCurrentExprProto()->mutable_literal_access_expr();

  protoExpr->set_value(expr->getValue());
  setBuiltinType(protoExpr->mutable_type(), expr->getBuiltinType());

  setLocation(protoExpr->mutable_loc(), expr->getSourceLocation());
  protoExpr->set_id(expr->getID());
}

template <typename DataTraits>
std::shared_ptr<Expr<DataTraits>> makeExpr(const proto::statements::Expr& expressionProto) {
  switch(expressionProto.expr_case()) {
  case proto::statements::Expr::kUnaryOperator: {
    const auto& exprProto = expressionProto.unary_operator();
    auto expr = std::make_shared<UnaryOperator<DataTraits>>(
        makeExpr<DataTraits>(exprProto.operand()), exprProto.op(), makeLocation(exprProto));
    expr->setID(exprProto.id());
    return expr;
  }
  case proto::statements::Expr::kBinaryOperator: {
    const auto& exprProto = expressionProto.binary_operator();
    auto expr = std::make_shared<BinaryOperator<DataTraits>>(
        makeExpr<DataTraits>(exprProto.left()), exprProto.op(),
        makeExpr<DataTraits>(exprProto.right()), makeLocation(exprProto));
    expr->setID(exprProto.id());
    return expr;
  }
  case proto::statements::Expr::kAssignmentExpr: {
    const auto& exprProto = expressionProto.assignment_expr();
    auto expr = std::make_shared<AssignmentExpr<DataTraits>>(
        makeExpr<DataTraits>(exprProto.left()), makeExpr<DataTraits>(exprProto.right()),
        exprProto.op(), makeLocation(exprProto));
    expr->setID(exprProto.id());
    return expr;
  }
  case proto::statements::Expr::kTernaryOperator: {
    const auto& exprProto = expressionProto.ternary_operator();
    auto expr = std::make_shared<TernaryOperator<DataTraits>>(
        makeExpr<DataTraits>(exprProto.cond()), makeExpr<DataTraits>(exprProto.left()),
        makeExpr<DataTraits>(exprProto.right()), makeLocation(exprProto));
    expr->setID(exprProto.id());
    return expr;
  }
  case proto::statements::Expr::kFunCallExpr: {
    const auto& exprProto = expressionProto.fun_call_expr();
    auto expr =
        std::make_shared<FunCallExpr<DataTraits>>(exprProto.callee(), makeLocation(exprProto));
    for(const auto& argProto : exprProto.arguments())
      expr->getArguments().emplace_back(makeExpr<DataTraits>(argProto));
    expr->setID(exprProto.id());
    return expr;
  }
  case proto::statements::Expr::kStencilFunCallExpr: {
    const auto& exprProto = expressionProto.stencil_fun_call_expr();
    auto expr = std::make_shared<StencilFunCallExpr<DataTraits>>(exprProto.callee(),
                                                                 makeLocation(exprProto));
    for(const auto& argProto : exprProto.arguments())
      expr->getArguments().emplace_back(makeExpr<DataTraits>(argProto));
    expr->setID(exprProto.id());
    return expr;
  }
  case proto::statements::Expr::kStencilFunArgExpr: {
    const auto& exprProto = expressionProto.stencil_fun_arg_expr();
    int direction = -1, offset = 0, argumentIndex = -1; // default values

    if(exprProto.has_dimension()) {
      switch(exprProto.dimension().direction()) {
      case proto::statements::Dimension_Direction_I:
        direction = 0;
        break;
      case proto::statements::Dimension_Direction_J:
        direction = 1;
        break;
      case proto::statements::Dimension_Direction_K:
        direction = 2;
        break;
      case proto::statements::Dimension_Direction_Invalid:
      default:
        direction = -1;
        break;
      }
    }
    offset = exprProto.offset();
    argumentIndex = exprProto.argument_index();
    auto expr = std::make_shared<StencilFunArgExpr<DataTraits>>(direction, offset, argumentIndex,
                                                                makeLocation(exprProto));
    expr->setID(exprProto.id());
    return expr;
  }
  case proto::statements::Expr::kVarAccessExpr: {
    const auto& exprProto = expressionProto.var_access_expr();
    auto expr = std::make_shared<VarAccessExpr<DataTraits>>(
        exprProto.name(), exprProto.has_index() ? makeExpr<DataTraits>(exprProto.index()) : nullptr,
        makeLocation(exprProto));
    expr->setIsExternal(exprProto.is_external());
    expr->setID(exprProto.id());
    return expr;
  }
  case proto::statements::Expr::kFieldAccessExpr: {
    const auto& exprProto = expressionProto.field_access_expr();
    auto name = exprProto.name();
    auto negateOffset = exprProto.negate_offset();

    auto throwException = [&exprProto](const char* member) {
      throw std::runtime_error(format("FieldAccessExpr::%s (loc %s) exceeds 3 dimensions", member,
                                      makeLocation(exprProto)));
    };

    Array3i offset{{0, 0, 0}};
    if(!exprProto.offset().empty()) {
      if(exprProto.offset().size() > 3)
        throwException("offset");

      std::copy(exprProto.offset().begin(), exprProto.offset().end(), offset.begin());
    }

    Array3i argumentOffset{{0, 0, 0}};
    if(!exprProto.argument_offset().empty()) {
      if(exprProto.argument_offset().size() > 3)
        throwException("argument_offset");

      std::copy(exprProto.argument_offset().begin(), exprProto.argument_offset().end(),
                argumentOffset.begin());
    }

    Array3i argumentMap{{-1, -1, -1}};
    if(!exprProto.argument_map().empty()) {
      if(exprProto.argument_map().size() > 3)
        throwException("argument_map");

      std::copy(exprProto.argument_map().begin(), exprProto.argument_map().end(),
                argumentMap.begin());
    }

    auto expr = std::make_shared<FieldAccessExpr<DataTraits>>(
        name, offset, argumentMap, argumentOffset, negateOffset, makeLocation(exprProto));
    expr->setID(exprProto.id());
    return expr;
  }
  case proto::statements::Expr::kLiteralAccessExpr: {
    const auto& exprProto = expressionProto.literal_access_expr();
    auto expr = std::make_shared<LiteralAccessExpr<DataTraits>>(
        exprProto.value(), makeBuiltinTypeID(exprProto.type()), makeLocation(exprProto));
    expr->setID(exprProto.id());
    return expr;
  }
  case proto::statements::Expr::EXPR_NOT_SET:
  default:
    dawn_unreachable("expr not set");
  }
  return nullptr;
}

template <typename DataTraits>
std::shared_ptr<AST<DataTraits>> makeAST(const dawn::proto::statements::AST& astProto) {
  auto ast = std::make_shared<AST<DataTraits>>();
  auto root = dyn_pointer_cast<BlockStmt<DataTraits>>(makeStmt<DataTraits>(astProto.root()));
  if(!root)
    throw std::runtime_error("root statement of AST is not a 'BlockStmt'");
  ast->setRoot(root);
  return ast;
}
