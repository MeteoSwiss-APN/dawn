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

#include "dawn/Serialization/ASTSerializer.h"
#include <fstream>
#include <google/protobuf/util/json_util.h>
#include <list>
#include <tuple>
#include <utility>

using namespace dawn;
using namespace ast;

void setAST(dawn::proto::statements::AST* astProto, const AST* ast);

void setLocation(dawn::proto::statements::SourceLocation* locProto, const SourceLocation& loc) {
  locProto->set_column(loc.Column);
  locProto->set_line(loc.Line);
}

void setBuiltinType(dawn::proto::statements::BuiltinType* builtinTypeProto,
                    const BuiltinTypeID& builtinType) {
  builtinTypeProto->set_type_id(
      static_cast<dawn::proto::statements::BuiltinType_TypeID>(builtinType));
}

void setInterval(dawn::proto::statements::Interval* intervalProto, const sir::Interval* interval) {
  if(interval->LowerLevel == sir::Interval::Start)
    intervalProto->set_special_lower_level(dawn::proto::statements::Interval::Start);
  else if(interval->LowerLevel == sir::Interval::End)
    intervalProto->set_special_lower_level(dawn::proto::statements::Interval::End);
  else
    intervalProto->set_lower_level(interval->LowerLevel);

  if(interval->UpperLevel == sir::Interval::Start)
    intervalProto->set_special_upper_level(dawn::proto::statements::Interval::Start);
  else if(interval->UpperLevel == sir::Interval::End)
    intervalProto->set_special_upper_level(dawn::proto::statements::Interval::End);
  else
    intervalProto->set_upper_level(interval->UpperLevel);

  intervalProto->set_lower_offset(interval->LowerOffset);
  intervalProto->set_upper_offset(interval->UpperOffset);
}

void setDirection(dawn::proto::statements::Direction* directionProto,
                  const sir::Direction* direction) {
  directionProto->set_name(direction->Name);
  setLocation(directionProto->mutable_loc(), direction->Loc);
}

void setOffset(dawn::proto::statements::Offset* offsetProto, const sir::Offset* offset) {
  offsetProto->set_name(offset->Name);
  setLocation(offsetProto->mutable_loc(), offset->Loc);
}

void setField(dawn::proto::statements::Field* fieldProto, const sir::Field* field) {
  fieldProto->set_name(field->Name);
  fieldProto->set_is_temporary(field->IsTemporary);
  for(const auto& initializedDimension : field->fieldDimensions) {
    fieldProto->add_field_dimensions(initializedDimension);
  }
  setLocation(fieldProto->mutable_loc(), field->Loc);
}

ProtoStmtBuilder::ProtoStmtBuilder(dawn::proto::statements::Stmt* stmtProto) {
  currentStmtProto_.push(stmtProto);
}

ProtoStmtBuilder::ProtoStmtBuilder(dawn::proto::statements::Expr* exprProto) {
  currentExprProto_.push(exprProto);
}

dawn::proto::statements::Stmt* ProtoStmtBuilder::getCurrentStmtProto() {
  DAWN_ASSERT(!currentStmtProto_.empty());
  return currentStmtProto_.top();
}

dawn::proto::statements::Expr* ProtoStmtBuilder::getCurrentExprProto() {
  DAWN_ASSERT(!currentExprProto_.empty());
  return currentExprProto_.top();
}
void ProtoStmtBuilder::visit(const std::shared_ptr<ReductionOverNeighborExpr>& expr) {
  DAWN_ASSERT_MSG(0, "Not implemented!");
}

void ProtoStmtBuilder::visit(const std::shared_ptr<BlockStmt>& stmt) {
  auto protoStmt = getCurrentStmtProto()->mutable_block_stmt();

  for(const auto& s : stmt->getStatements()) {
    currentStmtProto_.push(protoStmt->add_statements());
    s->accept(*this);
    currentStmtProto_.pop();
  }

  setLocation(protoStmt->mutable_loc(), stmt->getSourceLocation());
  protoStmt->set_id(stmt->getID());
}

void ProtoStmtBuilder::visit(const std::shared_ptr<ExprStmt>& stmt) {
  auto protoStmt = getCurrentStmtProto()->mutable_expr_stmt();
  currentExprProto_.push(protoStmt->mutable_expr());
  stmt->getExpr()->accept(*this);
  currentExprProto_.pop();

  setLocation(protoStmt->mutable_loc(), stmt->getSourceLocation());
  protoStmt->set_id(stmt->getID());
}

void ProtoStmtBuilder::visit(const std::shared_ptr<ReturnStmt>& stmt) {
  auto protoStmt = getCurrentStmtProto()->mutable_return_stmt();

  currentExprProto_.push(protoStmt->mutable_expr());
  stmt->getExpr()->accept(*this);
  currentExprProto_.pop();

  setLocation(protoStmt->mutable_loc(), stmt->getSourceLocation());
  protoStmt->set_id(stmt->getID());
}

void ProtoStmtBuilder::visit(const std::shared_ptr<VarDeclStmt>& stmt) {
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

void ProtoStmtBuilder::visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) {
  auto protoStmt = getCurrentStmtProto()->mutable_vertical_region_decl_stmt();

  dawn::sir::VerticalRegion* verticalRegion = stmt->getVerticalRegion().get();
  dawn::proto::statements::VerticalRegion* verticalRegionProto =
      protoStmt->mutable_vertical_region();

  // VerticalRegion.Loc
  setLocation(verticalRegionProto->mutable_loc(), verticalRegion->Loc);

  // VerticalRegion.Ast
  setAST(verticalRegionProto->mutable_ast(), verticalRegion->Ast.get());

  // VerticalRegion.VerticalInterval
  setInterval(verticalRegionProto->mutable_interval(), verticalRegion->VerticalInterval.get());

  // VerticalRegion.LoopOrder
  verticalRegionProto->set_loop_order(verticalRegion->LoopOrder ==
                                              dawn::sir::VerticalRegion::LK_Backward
                                          ? dawn::proto::statements::VerticalRegion::Backward
                                          : dawn::proto::statements::VerticalRegion::Forward);

  setLocation(protoStmt->mutable_loc(), stmt->getSourceLocation());
  protoStmt->set_id(stmt->getID());
}

void ProtoStmtBuilder::visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) {
  auto protoStmt = getCurrentStmtProto()->mutable_stencil_call_decl_stmt();

  dawn::sir::StencilCall* stencilCall = stmt->getStencilCall().get();
  dawn::proto::statements::StencilCall* stencilCallProto = protoStmt->mutable_stencil_call();

  // StencilCall.Loc
  setLocation(stencilCallProto->mutable_loc(), stencilCall->Loc);

  // StencilCall.Callee
  stencilCallProto->set_callee(stencilCall->Callee);

  // StencilCall.Args
  for(const auto& arg : stencilCall->Args) {
    auto argProto = stencilCallProto->add_arguments();
    argProto->set_name(arg->Name);
    argProto->set_is_temporary(arg->IsTemporary);
    argProto->mutable_loc()->set_column(arg->Loc.Column);
    argProto->mutable_loc()->set_line(arg->Loc.Line);
  }

  setLocation(protoStmt->mutable_loc(), stmt->getSourceLocation());
  protoStmt->set_id(stmt->getID());
}

void ProtoStmtBuilder::visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) {
  auto protoStmt = getCurrentStmtProto()->mutable_boundary_condition_decl_stmt();
  protoStmt->set_functor(stmt->getFunctor());

  for(const auto& field : stmt->getFields()) {
    auto fieldProto = protoStmt->add_fields();
    fieldProto->set_name(field->Name);
    fieldProto->set_is_temporary(field->IsTemporary);
    fieldProto->mutable_loc()->set_column(field->Loc.Column);
    fieldProto->mutable_loc()->set_line(field->Loc.Line);
  }

  setLocation(protoStmt->mutable_loc(), stmt->getSourceLocation());
  protoStmt->set_id(stmt->getID());
}

void ProtoStmtBuilder::visit(const std::shared_ptr<IfStmt>& stmt) {
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

void ProtoStmtBuilder::visit(const std::shared_ptr<UnaryOperator>& expr) {
  auto protoExpr = getCurrentExprProto()->mutable_unary_operator();
  protoExpr->set_op(expr->getOp());

  currentExprProto_.push(protoExpr->mutable_operand());
  expr->getOperand()->accept(*this);
  currentExprProto_.pop();

  setLocation(protoExpr->mutable_loc(), expr->getSourceLocation());
  protoExpr->set_id(expr->getID());
}

void ProtoStmtBuilder::visit(const std::shared_ptr<BinaryOperator>& expr) {
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

void ProtoStmtBuilder::visit(const std::shared_ptr<AssignmentExpr>& expr) {
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

void ProtoStmtBuilder::visit(const std::shared_ptr<TernaryOperator>& expr) {
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

void ProtoStmtBuilder::visit(const std::shared_ptr<FunCallExpr>& expr) {
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

void ProtoStmtBuilder::visit(const std::shared_ptr<StencilFunCallExpr>& expr) {
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

void ProtoStmtBuilder::visit(const std::shared_ptr<StencilFunArgExpr>& expr) {
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

void ProtoStmtBuilder::visit(const std::shared_ptr<VarAccessExpr>& expr) {
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

void ProtoStmtBuilder::visit(const std::shared_ptr<FieldAccessExpr>& expr) {
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

void ProtoStmtBuilder::visit(const std::shared_ptr<LiteralAccessExpr>& expr) {
  auto protoExpr = getCurrentExprProto()->mutable_literal_access_expr();

  protoExpr->set_value(expr->getValue());
  setBuiltinType(protoExpr->mutable_type(), expr->getBuiltinType());

  setLocation(protoExpr->mutable_loc(), expr->getSourceLocation());
  protoExpr->set_id(expr->getID());
}

void setAST(proto::statements::AST* astProto, const AST* ast) {
  ProtoStmtBuilder builder(astProto->mutable_root());
  ast->accept(builder);
}

//===------------------------------------------------------------------------------------------===//
// Deserialization
//===------------------------------------------------------------------------------------------===//

std::shared_ptr<sir::Field> makeField(const proto::statements::Field& fieldProto) {
  auto field = std::make_shared<sir::Field>(fieldProto.name(), makeLocation(fieldProto));
  field->IsTemporary = fieldProto.is_temporary();
  if(!fieldProto.field_dimensions().empty()) {
    auto throwException = [&fieldProto](const char* member) {
      throw std::runtime_error(
          format("Field::%s (loc %s) exceeds 3 dimensions", member, makeLocation(fieldProto)));
    };
    if(fieldProto.field_dimensions().size() > 3)
      throwException("field_dimensions");

    std::copy(fieldProto.field_dimensions().begin(), fieldProto.field_dimensions().end(),
              field->fieldDimensions.begin());
  }
  return field;
}

BuiltinTypeID makeBuiltinTypeID(const proto::statements::BuiltinType& builtinTypeProto) {
  switch(builtinTypeProto.type_id()) {
  case proto::statements::BuiltinType_TypeID_Invalid:
    return BuiltinTypeID::Invalid;
  case proto::statements::BuiltinType_TypeID_Auto:
    return BuiltinTypeID::Auto;
  case proto::statements::BuiltinType_TypeID_Boolean:
    return BuiltinTypeID::Boolean;
  case proto::statements::BuiltinType_TypeID_Integer:
    return BuiltinTypeID::Integer;
  case proto::statements::BuiltinType_TypeID_Float:
    return BuiltinTypeID::Float;
  default:
    return BuiltinTypeID::Invalid;
  }
  return BuiltinTypeID::Invalid;
}

std::shared_ptr<sir::Direction> makeDirection(const proto::statements::Direction& directionProto) {
  return std::make_shared<sir::Direction>(directionProto.name(), makeLocation(directionProto));
}

std::shared_ptr<sir::Offset> makeOffset(const proto::statements::Offset& offsetProto) {
  return std::make_shared<sir::Offset>(offsetProto.name(), makeLocation(offsetProto));
}

std::shared_ptr<sir::Interval> makeInterval(const proto::statements::Interval& intervalProto) {
  int lowerLevel = -1, upperLevel = -1, lowerOffset = -1, upperOffset = -1;

  if(intervalProto.LowerLevel_case() == proto::statements::Interval::kSpecialLowerLevel)
    lowerLevel = intervalProto.special_lower_level() ==
                         proto::statements::Interval_SpecialLevel::Interval_SpecialLevel_Start
                     ? sir::Interval::Start
                     : sir::Interval::End;
  else
    lowerLevel = intervalProto.lower_level();

  if(intervalProto.UpperLevel_case() == proto::statements::Interval::kSpecialUpperLevel)
    upperLevel = intervalProto.special_upper_level() ==
                         proto::statements::Interval_SpecialLevel::Interval_SpecialLevel_Start
                     ? sir::Interval::Start
                     : sir::Interval::End;
  else
    upperLevel = intervalProto.upper_level();

  lowerOffset = intervalProto.lower_offset();
  upperOffset = intervalProto.upper_offset();
  return std::make_shared<sir::Interval>(lowerLevel, upperLevel, lowerOffset, upperOffset);
}

std::shared_ptr<Expr> makeExpr(const proto::statements::Expr& expressionProto) {
  switch(expressionProto.expr_case()) {
  case proto::statements::Expr::kUnaryOperator: {
    const auto& exprProto = expressionProto.unary_operator();
    auto expr = std::make_shared<UnaryOperator>(makeExpr(exprProto.operand()), exprProto.op(),
                                                makeLocation(exprProto));
    expr->setID(exprProto.id());
    return expr;
  }
  case proto::statements::Expr::kBinaryOperator: {
    const auto& exprProto = expressionProto.binary_operator();
    auto expr =
        std::make_shared<BinaryOperator>(makeExpr(exprProto.left()), exprProto.op(),
                                         makeExpr(exprProto.right()), makeLocation(exprProto));
    expr->setID(exprProto.id());
    return expr;
  }
  case proto::statements::Expr::kAssignmentExpr: {
    const auto& exprProto = expressionProto.assignment_expr();
    auto expr =
        std::make_shared<AssignmentExpr>(makeExpr(exprProto.left()), makeExpr(exprProto.right()),
                                         exprProto.op(), makeLocation(exprProto));
    expr->setID(exprProto.id());
    return expr;
  }
  case proto::statements::Expr::kTernaryOperator: {
    const auto& exprProto = expressionProto.ternary_operator();
    auto expr =
        std::make_shared<TernaryOperator>(makeExpr(exprProto.cond()), makeExpr(exprProto.left()),
                                          makeExpr(exprProto.right()), makeLocation(exprProto));
    expr->setID(exprProto.id());
    return expr;
  }
  case proto::statements::Expr::kFunCallExpr: {
    const auto& exprProto = expressionProto.fun_call_expr();
    auto expr = std::make_shared<FunCallExpr>(exprProto.callee(), makeLocation(exprProto));
    for(const auto& argProto : exprProto.arguments())
      expr->getArguments().emplace_back(makeExpr(argProto));
    expr->setID(exprProto.id());
    return expr;
  }
  case proto::statements::Expr::kStencilFunCallExpr: {
    const auto& exprProto = expressionProto.stencil_fun_call_expr();
    auto expr = std::make_shared<StencilFunCallExpr>(exprProto.callee(), makeLocation(exprProto));
    for(const auto& argProto : exprProto.arguments())
      expr->getArguments().emplace_back(makeExpr(argProto));
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
    auto expr = std::make_shared<StencilFunArgExpr>(direction, offset, argumentIndex,
                                                    makeLocation(exprProto));
    expr->setID(exprProto.id());
    return expr;
  }
  case proto::statements::Expr::kVarAccessExpr: {
    const auto& exprProto = expressionProto.var_access_expr();
    auto expr = std::make_shared<VarAccessExpr>(
        exprProto.name(), exprProto.has_index() ? makeExpr(exprProto.index()) : nullptr,
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

    auto expr = std::make_shared<FieldAccessExpr>(name, offset, argumentMap, argumentOffset,
                                                  negateOffset, makeLocation(exprProto));
    expr->setID(exprProto.id());
    return expr;
  }
  case proto::statements::Expr::kLiteralAccessExpr: {
    const auto& exprProto = expressionProto.literal_access_expr();
    auto expr = std::make_shared<LiteralAccessExpr>(
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

std::shared_ptr<Stmt> makeStmt(const proto::statements::Stmt& statementProto) {
  switch(statementProto.stmt_case()) {
  case proto::statements::Stmt::kBlockStmt: {
    const auto& stmtProto = statementProto.block_stmt();
    auto stmt = std::make_shared<BlockStmt>(makeLocation(stmtProto));

    for(const auto& s : stmtProto.statements())
      stmt->push_back(makeStmt(s));
    stmt->setID(stmtProto.id());

    return stmt;
  }
  case proto::statements::Stmt::kExprStmt: {
    const auto& stmtProto = statementProto.expr_stmt();
    auto stmt = std::make_shared<ExprStmt>(makeExpr(stmtProto.expr()), makeLocation(stmtProto));
    stmt->setID(stmtProto.id());
    return stmt;
  }
  case proto::statements::Stmt::kReturnStmt: {
    const auto& stmtProto = statementProto.return_stmt();
    auto stmt = std::make_shared<ReturnStmt>(makeExpr(stmtProto.expr()), makeLocation(stmtProto));
    stmt->setID(stmtProto.id());
    return stmt;
  }
  case proto::statements::Stmt::kVarDeclStmt: {
    const auto& stmtProto = statementProto.var_decl_stmt();

    std::vector<std::shared_ptr<Expr>> initList;
    for(const auto& e : stmtProto.init_list())
      initList.emplace_back(makeExpr(e));

    const proto::statements::Type& typeProto = stmtProto.type();
    CVQualifier cvQual = CVQualifier::Invalid;
    if(typeProto.is_const())
      cvQual |= CVQualifier::Const;
    if(typeProto.is_volatile())
      cvQual |= CVQualifier::Volatile;
    Type type = typeProto.name().empty() ? Type(makeBuiltinTypeID(typeProto.builtin_type()), cvQual)
                                         : Type(typeProto.name(), cvQual);

    auto stmt =
        std::make_shared<VarDeclStmt>(type, stmtProto.name(), stmtProto.dimension(),
                                      stmtProto.op().c_str(), initList, makeLocation(stmtProto));
    stmt->setID(stmtProto.id());
    return stmt;
  }
  case proto::statements::Stmt::kStencilCallDeclStmt: {
    auto metaloc = makeLocation(statementProto.stencil_call_decl_stmt());
    const auto& stmtProto = statementProto.stencil_call_decl_stmt();
    auto loc = makeLocation(stmtProto.stencil_call());
    std::shared_ptr<sir::StencilCall> call =
        std::make_shared<sir::StencilCall>(stmtProto.stencil_call().callee(), loc);
    for(const auto& arg : stmtProto.stencil_call().arguments()) {
      call->Args.push_back(makeField(arg));
    }
    auto stmt = std::make_shared<StencilCallDeclStmt>(call, metaloc);
    stmt->setID(stmtProto.id());
    return stmt;
  }
  case proto::statements::Stmt::kVerticalRegionDeclStmt: {
    const auto& stmtProto = statementProto.vertical_region_decl_stmt();
    auto loc = makeLocation(stmtProto.vertical_region());
    std::shared_ptr<sir::Interval> interval = makeInterval(stmtProto.vertical_region().interval());
    sir::VerticalRegion::LoopOrderKind looporder;
    switch(stmtProto.vertical_region().loop_order()) {
    case proto::statements::VerticalRegion_LoopOrder_Forward:
      looporder = sir::VerticalRegion::LK_Forward;
      break;
    case proto::statements::VerticalRegion_LoopOrder_Backward:
      looporder = sir::VerticalRegion::LK_Backward;
      break;
    default:
      dawn_unreachable("no looporder specified");
    }
    auto ast = makeAST(stmtProto.vertical_region().ast());
    std::shared_ptr<sir::VerticalRegion> verticalRegion =
        std::make_shared<sir::VerticalRegion>(ast, interval, looporder, loc);
    auto stmt = std::make_shared<VerticalRegionDeclStmt>(verticalRegion, loc);
    stmt->setID(stmtProto.id());
    return stmt;
  }
  case proto::statements::Stmt::kBoundaryConditionDeclStmt: {
    const auto& stmtProto = statementProto.boundary_condition_decl_stmt();
    auto stmt =
        std::make_shared<BoundaryConditionDeclStmt>(stmtProto.functor(), makeLocation(stmtProto));
    for(const auto& fieldProto : stmtProto.fields())
      stmt->getFields().emplace_back(makeField(fieldProto));
    stmt->setID(stmtProto.id());
    return stmt;
  }
  case proto::statements::Stmt::kIfStmt: {
    const auto& stmtProto = statementProto.if_stmt();
    auto stmt = std::make_shared<IfStmt>(
        makeStmt(stmtProto.cond_part()), makeStmt(stmtProto.then_part()),
        stmtProto.has_else_part() ? makeStmt(stmtProto.else_part()) : nullptr,
        makeLocation(stmtProto));
    stmt->setID(stmtProto.id());
    return stmt;
  }
  case proto::statements::Stmt::STMT_NOT_SET:
  default:
    dawn_unreachable("stmt not set");
  }
  return nullptr;
}

std::shared_ptr<AST> makeAST(const dawn::proto::statements::AST& astProto) {
  auto ast = std::make_shared<AST>();
  auto root = dyn_pointer_cast<BlockStmt>(makeStmt(astProto.root()));
  if(!root)
    throw std::runtime_error("root statement of AST is not a 'BlockStmt'");
  ast->setRoot(root);
  return ast;
}
