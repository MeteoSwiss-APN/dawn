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

#include "dawn/IIR/IIRSerializer.h"
#include "dawn/IIR/IIR.pb.h"
#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include <fstream>
#include <google/protobuf/util/json_util.h>
#include <list>
#include <stack>
#include <tuple>
#include <utility>

using namespace dawn;

namespace {
///////////////////// Copy pasta, move to common

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

static void setAST(dawn::proto::statements::AST* astProto, const AST* ast);

static void setLocation(dawn::proto::statements::SourceLocation* locProto,
                        const SourceLocation& loc) {
  locProto->set_column(loc.Column);
  locProto->set_line(loc.Line);
}

static void setBuiltinType(dawn::proto::statements::BuiltinType* builtinTypeProto,
                           const BuiltinTypeID& builtinType) {
  builtinTypeProto->set_type_id(
      static_cast<dawn::proto::statements::BuiltinType_TypeID>(builtinType));
}

static void setInterval(dawn::proto::statements::Interval* intervalProto,
                        const sir::Interval* interval) {
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

class ProtoStmtBuilder : public dawn::ASTVisitor {
  std::stack<dawn::proto::statements::Stmt*> currentStmtProto_;
  std::stack<dawn::proto::statements::Expr*> currentExprProto_;

public:
  ProtoStmtBuilder(dawn::proto::statements::Stmt* stmtProto) { currentStmtProto_.push(stmtProto); }

  dawn::proto::statements::Stmt* getCurrentStmtProto() {
    DAWN_ASSERT(!currentStmtProto_.empty());
    return currentStmtProto_.top();
  }

  dawn::proto::statements::Expr* getCurrentExprProto() {
    DAWN_ASSERT(!currentExprProto_.empty());
    return currentExprProto_.top();
  }

  void visit(const std::shared_ptr<BlockStmt>& stmt) override {
    auto protoStmt = getCurrentStmtProto()->mutable_block_stmt();

    for(const auto& s : stmt->getStatements()) {
      currentStmtProto_.push(protoStmt->add_statements());
      s->accept(*this);
      currentStmtProto_.pop();
    }

    setLocation(protoStmt->mutable_loc(), stmt->getSourceLocation());
  }

  void visit(const std::shared_ptr<ExprStmt>& stmt) override {
    auto protoStmt = getCurrentStmtProto()->mutable_expr_stmt();
    currentExprProto_.push(protoStmt->mutable_expr());
    stmt->getExpr()->accept(*this);
    currentExprProto_.pop();

    setLocation(protoStmt->mutable_loc(), stmt->getSourceLocation());
  }

  void visit(const std::shared_ptr<ReturnStmt>& stmt) override {
    auto protoStmt = getCurrentStmtProto()->mutable_return_stmt();

    currentExprProto_.push(protoStmt->mutable_expr());
    stmt->getExpr()->accept(*this);
    currentExprProto_.pop();

    setLocation(protoStmt->mutable_loc(), stmt->getSourceLocation());
  }

  void visit(const std::shared_ptr<VarDeclStmt>& stmt) override {
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
  }

  void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) override {
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
  }

  void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) override {
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
  }

  void visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) override {
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
  }

  void visit(const std::shared_ptr<IfStmt>& stmt) override {
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
  }

  void visit(const std::shared_ptr<UnaryOperator>& expr) override {
    auto protoExpr = getCurrentExprProto()->mutable_unary_operator();
    protoExpr->set_op(expr->getOp());

    currentExprProto_.push(protoExpr->mutable_operand());
    expr->getOperand()->accept(*this);
    currentExprProto_.pop();

    setLocation(protoExpr->mutable_loc(), expr->getSourceLocation());
  }

  void visit(const std::shared_ptr<BinaryOperator>& expr) override {
    auto protoExpr = getCurrentExprProto()->mutable_binary_operator();
    protoExpr->set_op(expr->getOp());

    currentExprProto_.push(protoExpr->mutable_left());
    expr->getLeft()->accept(*this);
    currentExprProto_.pop();

    currentExprProto_.push(protoExpr->mutable_right());
    expr->getRight()->accept(*this);
    currentExprProto_.pop();

    setLocation(protoExpr->mutable_loc(), expr->getSourceLocation());
  }

  void visit(const std::shared_ptr<AssignmentExpr>& expr) override {
    auto protoExpr = getCurrentExprProto()->mutable_assignment_expr();
    protoExpr->set_op(expr->getOp());

    currentExprProto_.push(protoExpr->mutable_left());
    expr->getLeft()->accept(*this);
    currentExprProto_.pop();

    currentExprProto_.push(protoExpr->mutable_right());
    expr->getRight()->accept(*this);
    currentExprProto_.pop();

    setLocation(protoExpr->mutable_loc(), expr->getSourceLocation());
  }

  void visit(const std::shared_ptr<TernaryOperator>& expr) override {
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
  }

  void visit(const std::shared_ptr<FunCallExpr>& expr) override {
    auto protoExpr = getCurrentExprProto()->mutable_fun_call_expr();
    protoExpr->set_callee(expr->getCallee());

    for(const auto& arg : expr->getArguments()) {
      currentExprProto_.push(protoExpr->add_arguments());
      arg->accept(*this);
      currentExprProto_.pop();
    }

    setLocation(protoExpr->mutable_loc(), expr->getSourceLocation());
  }

  void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override {
    auto protoExpr = getCurrentExprProto()->mutable_stencil_fun_call_expr();
    protoExpr->set_callee(expr->getCallee());

    for(const auto& arg : expr->getArguments()) {
      currentExprProto_.push(protoExpr->add_arguments());
      arg->accept(*this);
      currentExprProto_.pop();
    }

    setLocation(protoExpr->mutable_loc(), expr->getSourceLocation());
  }

  void visit(const std::shared_ptr<StencilFunArgExpr>& expr) override {
    auto protoExpr = getCurrentExprProto()->mutable_stencil_fun_arg_expr();

    protoExpr->mutable_dimension()->set_direction(
        expr->getDimension() == -1
            ? dawn::proto::statements::Dimension::Invalid
            : static_cast<dawn::proto::statements::Dimension_Direction>(expr->getDimension()));
    protoExpr->set_offset(expr->getOffset());
    protoExpr->set_argument_index(expr->getArgumentIndex());

    setLocation(protoExpr->mutable_loc(), expr->getSourceLocation());
  }

  void visit(const std::shared_ptr<VarAccessExpr>& expr) override {
    auto protoExpr = getCurrentExprProto()->mutable_var_access_expr();

    protoExpr->set_name(expr->getName());
    protoExpr->set_is_external(expr->isExternal());

    if(expr->isArrayAccess()) {
      currentExprProto_.push(protoExpr->mutable_index());
      expr->getIndex()->accept(*this);
      currentExprProto_.pop();
    }

    setLocation(protoExpr->mutable_loc(), expr->getSourceLocation());
  }

  void visit(const std::shared_ptr<FieldAccessExpr>& expr) override {
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
  }

  void visit(const std::shared_ptr<LiteralAccessExpr>& expr) override {
    auto protoExpr = getCurrentExprProto()->mutable_literal_access_expr();

    protoExpr->set_value(expr->getValue());
    setBuiltinType(protoExpr->mutable_type(), expr->getBuiltinType());

    setLocation(protoExpr->mutable_loc(), expr->getSourceLocation());
  }
};

static void setAST(proto::statements::AST* astProto, const AST* ast) {
  ProtoStmtBuilder builder(astProto->mutable_root());
  ast->accept(builder);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Deserialization
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T>
static SourceLocation makeLocation(const T& proto) {
  return proto.has_loc() ? SourceLocation(proto.loc().line(), proto.loc().column())
                         : SourceLocation{};
}

static std::shared_ptr<sir::Field> makeField(const proto::statements::Field& fieldProto) {
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

static BuiltinTypeID makeBuiltinTypeID(const proto::statements::BuiltinType& builtinTypeProto) {
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

// static std::shared_ptr<sir::Direction> makeDirection(const proto::statements::Direction&
// directionProto) {
//  return std::make_shared<sir::Direction>(directionProto.name(), makeLocation(directionProto));
//}

// static std::shared_ptr<sir::Offset> makeOffset(const proto::statements::Offset& offsetProto) {
//  return std::make_shared<sir::Offset>(offsetProto.name(), makeLocation(offsetProto));
//}

// static std::shared_ptr<sir::Interval> makeInterval(const proto::statements::Interval&
// intervalProto) {
//  int lowerLevel = -1, upperLevel = -1, lowerOffset = -1, upperOffset = -1;

//  if(intervalProto.LowerLevel_case() == proto::statements::Interval::kSpecialLowerLevel)
//    lowerLevel = intervalProto.special_lower_level() ==
//                         proto::statements::Interval_SpecialLevel::Interval_SpecialLevel_Start
//                     ? sir::Interval::Start
//                     : sir::Interval::End;
//  else
//    lowerLevel = intervalProto.lower_level();

//  if(intervalProto.UpperLevel_case() == proto::statements::Interval::kSpecialUpperLevel)
//    upperLevel = intervalProto.special_upper_level() ==
//                         proto::statements::Interval_SpecialLevel::Interval_SpecialLevel_Start
//                     ? sir::Interval::Start
//                     : sir::Interval::End;
//  else
//    upperLevel = intervalProto.upper_level();

//  lowerOffset = intervalProto.lower_offset();
//  upperOffset = intervalProto.upper_offset();
//  return std::make_shared<sir::Interval>(lowerLevel, upperLevel, lowerOffset, upperOffset);
//}

static std::shared_ptr<Expr> makeExpr(const proto::statements::Expr& expressionProto) {
  switch(expressionProto.expr_case()) {
  case proto::statements::Expr::kUnaryOperator: {
    const auto& exprProto = expressionProto.unary_operator();
    return std::make_shared<UnaryOperator>(makeExpr(exprProto.operand()), exprProto.op(),
                                           makeLocation(exprProto));
  }
  case proto::statements::Expr::kBinaryOperator: {
    const auto& exprProto = expressionProto.binary_operator();
    return std::make_shared<BinaryOperator>(makeExpr(exprProto.left()), exprProto.op(),
                                            makeExpr(exprProto.right()), makeLocation(exprProto));
  }
  case proto::statements::Expr::kAssignmentExpr: {
    const auto& exprProto = expressionProto.assignment_expr();
    return std::make_shared<AssignmentExpr>(makeExpr(exprProto.left()), makeExpr(exprProto.right()),
                                            exprProto.op(), makeLocation(exprProto));
  }
  case proto::statements::Expr::kTernaryOperator: {
    const auto& exprProto = expressionProto.ternary_operator();
    return std::make_shared<TernaryOperator>(makeExpr(exprProto.cond()), makeExpr(exprProto.left()),
                                             makeExpr(exprProto.right()), makeLocation(exprProto));
  }
  case proto::statements::Expr::kFunCallExpr: {
    const auto& exprProto = expressionProto.fun_call_expr();
    auto expr = std::make_shared<FunCallExpr>(exprProto.callee(), makeLocation(exprProto));
    for(const auto& argProto : exprProto.arguments())
      expr->getArguments().emplace_back(makeExpr(argProto));
    return expr;
  }
  case proto::statements::Expr::kStencilFunCallExpr: {
    const auto& exprProto = expressionProto.stencil_fun_call_expr();
    auto expr = std::make_shared<StencilFunCallExpr>(exprProto.callee(), makeLocation(exprProto));
    for(const auto& argProto : exprProto.arguments())
      expr->getArguments().emplace_back(makeExpr(argProto));
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
    return std::make_shared<StencilFunArgExpr>(direction, offset, argumentIndex,
                                               makeLocation(exprProto));
  }
  case proto::statements::Expr::kVarAccessExpr: {
    const auto& exprProto = expressionProto.var_access_expr();
    auto expr = std::make_shared<VarAccessExpr>(
        exprProto.name(), exprProto.has_index() ? makeExpr(exprProto.index()) : nullptr,
        makeLocation(exprProto));
    expr->setIsExternal(exprProto.is_external());
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

    return std::make_shared<FieldAccessExpr>(name, offset, argumentMap, argumentOffset,
                                             negateOffset, makeLocation(exprProto));
  }
  case proto::statements::Expr::kLiteralAccessExpr: {
    const auto& exprProto = expressionProto.literal_access_expr();
    return std::make_shared<LiteralAccessExpr>(
        exprProto.value(), makeBuiltinTypeID(exprProto.type()), makeLocation(exprProto));
  }
  case proto::statements::Expr::EXPR_NOT_SET:
  default:
    dawn_unreachable("expr not set");
  }
  return nullptr;
}

static std::shared_ptr<Stmt> makeStmt(const proto::statements::Stmt& statementProto) {
  switch(statementProto.stmt_case()) {
  case proto::statements::Stmt::kBlockStmt: {
    const auto& stmtProto = statementProto.block_stmt();
    auto stmt = std::make_shared<BlockStmt>(makeLocation(stmtProto));

    for(const auto& s : stmtProto.statements())
      stmt->push_back(makeStmt(s));

    return stmt;
  }
  case proto::statements::Stmt::kExprStmt: {
    const auto& stmtProto = statementProto.expr_stmt();
    return std::make_shared<ExprStmt>(makeExpr(stmtProto.expr()), makeLocation(stmtProto));
  }
  case proto::statements::Stmt::kReturnStmt: {
    const auto& stmtProto = statementProto.return_stmt();
    return std::make_shared<ReturnStmt>(makeExpr(stmtProto.expr()), makeLocation(stmtProto));
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

    return std::make_shared<VarDeclStmt>(type, stmtProto.name(), stmtProto.dimension(),
                                         stmtProto.op().c_str(), initList, makeLocation(stmtProto));
  }
  case proto::statements::Stmt::kStencilCallDeclStmt: {
    DAWN_ASSERT_MSG(false, "Vertical Region not allowed in this context");
    return nullptr;
  }
  case proto::statements::Stmt::kVerticalRegionDeclStmt: {
    DAWN_ASSERT_MSG(false, "Vertical Region not allowed in this context");
    return nullptr;
  }
  case proto::statements::Stmt::kBoundaryConditionDeclStmt: {
    const auto& stmtProto = statementProto.boundary_condition_decl_stmt();
    auto stmt =
        std::make_shared<BoundaryConditionDeclStmt>(stmtProto.functor(), makeLocation(stmtProto));
    for(const auto& fieldProto : stmtProto.fields())
      stmt->getFields().emplace_back(makeField(fieldProto));
    return stmt;
  }
  case proto::statements::Stmt::kIfStmt: {
    const auto& stmtProto = statementProto.if_stmt();
    return std::make_shared<IfStmt>(
        makeStmt(stmtProto.cond_part()), makeStmt(stmtProto.then_part()),
        stmtProto.has_else_part() ? makeStmt(stmtProto.else_part()) : nullptr,
        makeLocation(stmtProto));
  }
  case proto::statements::Stmt::STMT_NOT_SET:
  default:
    dawn_unreachable("stmt not set");
  }
  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

void setAccesses(proto::iir::Acesses* protoAcesses,
                 const std::shared_ptr<iir::Accesses>& accesses) {
  auto protoReadAccesses = protoAcesses->mutable_readaccess();
  for(auto IDExtentsPair : accesses->getReadAccesses()) {
    proto::iir::Extents protoExtents;
    iir::Extents e = IDExtentsPair.second;
    for(auto extent : e.getExtents()) {
      auto protoExtent = protoExtents.add_extents();
      protoExtent->set_minus(extent.Minus);
      protoExtent->set_plus(extent.Plus);
    }
    protoReadAccesses->insert({IDExtentsPair.first, protoExtents});
  }

  auto protoWriteAccesses = protoAcesses->mutable_writeaccess();
  for(auto IDExtentsPair : accesses->getWriteAccesses()) {
    proto::iir::Extents protoExtents;
    iir::Extents e = IDExtentsPair.second;
    for(auto extent : e.getExtents()) {
      auto protoExtent = protoExtents.add_extents();
      protoExtent->set_minus(extent.Minus);
      protoExtent->set_plus(extent.Plus);
    }
    protoWriteAccesses->insert({IDExtentsPair.first, protoExtents});
  }
}

std::shared_ptr<dawn::Statement>
makeStatement(const proto::iir::StencilDescStatement* protoStatement) {
  auto stmt = makeStmt(protoStatement->stmt());
  // WITTODO: add stack trace here
  return std::make_shared<Statement>(stmt, nullptr);
}

void serializeStmtAccessPair(proto::iir::StatementAcessPair* protoStmtAccessPair,
                             const std::unique_ptr<iir::StatementAccessesPair>& stmtAccessPair) {
  // serialize the statement
  ProtoStmtBuilder builder(protoStmtAccessPair->mutable_statement()->mutable_aststmt());
  stmtAccessPair->getStatement()->ASTStmt->accept(builder);

  // TODO: I don't think this is actually needed...
  // check if caller / callee acesses are initialized, and if so, fill them
  if(stmtAccessPair->getCallerAccesses()) {
    setAccesses(protoStmtAccessPair->mutable_calleraccesses(), stmtAccessPair->getCallerAccesses());
  }
  if(stmtAccessPair->getCalleeAccesses()) {
    setAccesses(protoStmtAccessPair->mutable_calleeaccesses(), stmtAccessPair->getCalleeAccesses());
  }
}
} // anonymous namespace

void IIRSerializer::serializeMetaData(proto::iir::StencilInstantiation& target,
                                      iir::StencilMetaInformation& metaData) {
  auto protoMetaData = target.mutable_metadata();
  // Filling Field: map<int32, string> AccessIDToName = 1;
  auto& protoAccessIDtoNameMap = *protoMetaData->mutable_accessidtoname();
  for(const auto& accessIDtoNamePair : metaData.AccessIDToNameMap_) {
    protoAccessIDtoNameMap.insert({accessIDtoNamePair.first, accessIDtoNamePair.second});
  }
  // Filling Field: repeated ExprIDPair ExprToAccessID = 2;
  for(const auto& exprToAccessIDPair : metaData.ExprToAccessIDMap_) {
    auto protoExprToAccessID = protoMetaData->add_exprtoaccessid();
    //    ProtoStmtBuilder builder(protoExprToAccessID->mutable_expr());
    //    exprToAccessIDPair.first->accept(builder);
    protoExprToAccessID->set_ids(exprToAccessIDPair.second);
  }
  // Filling Field: repeated StmtIDPair StmtToAccessID = 3;
  for(const auto& stmtToAccessIDPair : metaData.StmtToAccessIDMap_) {
    auto protoStmtToAccessID = protoMetaData->add_stmttoaccessid();
    ProtoStmtBuilder builder(protoStmtToAccessID->mutable_stmt());
    stmtToAccessIDPair.first->accept(builder);
    protoStmtToAccessID->set_ids(stmtToAccessIDPair.second);
  }
  // Filling Field: map<int32, string> LiteralIDToName = 4;
  auto& protoLiteralIDToNameMap = *protoMetaData->mutable_literalidtoname();
  for(const auto& literalIDtoNamePair : metaData.LiteralAccessIDToNameMap_) {
    protoLiteralIDToNameMap.insert({literalIDtoNamePair.first, literalIDtoNamePair.second});
  }
  // Filling Field: repeated int32 FieldAccessIDs = 5;
  for(int fieldAccessID : metaData.FieldAccessIDSet_) {
    protoMetaData->add_fieldaccessids(fieldAccessID);
  }
  // Filling Field: repeated int32 APIFieldIDs = 6;
  for(int apifieldID : metaData.apiFieldIDs_) {
    protoMetaData->add_apifieldids(apifieldID);
  }
  // Filling Field: repeated int32 TemporaryFieldIDs = 7;
  for(int temporaryFieldID : metaData.TemporaryFieldAccessIDSet_) {
    protoMetaData->add_temporaryfieldids(temporaryFieldID);
  }
  // Filling Field: repeated int32 GlobalVariableIDs = 8;
  for(int globalVariableID : metaData.GlobalVariableAccessIDSet_) {
    protoMetaData->add_globalvariableids(globalVariableID);
  }

  // Filling Field: VariableVersions versionedFields = 9;
  auto protoVariableVersions = protoMetaData->mutable_versionedfields();
  auto protoVariableVersionMap = *protoVariableVersions->mutable_veriableversionmap();
  auto protoVersionIDtoOriginalIDMap = *protoVariableVersions->mutable_versionidtooriginalid();

  auto variableVersions = metaData.variableVersions_;
  for(int versionedID : variableVersions.getVersionIDs()) {
    protoVariableVersions->add_versionids(versionedID);
  }
  for(auto& IDtoVectorOfVersionsPair : variableVersions.variableVersionsMap_) {
    proto::iir::AllVersionedFields protoFieldVersions;
    for(int id : *(IDtoVectorOfVersionsPair.second)) {
      protoFieldVersions.add_allids(id);
    }
    protoVariableVersionMap.insert({IDtoVectorOfVersionsPair.first, protoFieldVersions});
  }
  for(auto& VersionedIDToOriginalID : variableVersions.versionToOriginalVersionMap_) {
    protoVersionIDtoOriginalIDMap.insert(
        {VersionedIDToOriginalID.first, VersionedIDToOriginalID.second});
  }
  // Filling Field: repeated StencilDescStatement stencilDescStatements = 10;
  for(const auto& stencilDescStmt : metaData.stencilDescStatements_) {
    auto protoStmt = protoMetaData->add_stencildescstatements();
    ProtoStmtBuilder builder(protoStmt->mutable_stmt());
    stencilDescStmt->ASTStmt->accept(builder);
    if(stencilDescStmt->StackTrace)
      for(auto sirStackTrace : *(stencilDescStmt->StackTrace)) {
        auto protoStackTrace = protoStmt->add_stacktrace();
        setLocation(protoStackTrace->mutable_loc(), sirStackTrace->Loc);
        protoStackTrace->set_callee(sirStackTrace->Callee);
        for(auto argument : sirStackTrace->Args) {
          auto arg = protoStackTrace->add_arguments();
          arg->set_name(argument->Name);
          setLocation(arg->mutable_loc(), argument->Loc);
          arg->set_is_temporary(argument->IsTemporary);
          for(int dim : argument->fieldDimensions) {
            arg->add_field_dimensions(dim);
          }
        }
      }
  }
  // Filling Field: map<int32, dawn.proto.statements.StencilCallDeclStmt> IDToStencilCall = 11;

  // Filling Field:
  // map<string, dawn.proto.statements.BoundaryConditionDeclStmt> FieldnameToBoundaryCondition = 12;

  // Filling Field: map<int32, Array3i> fieldIDtoLegalDimensions = 13;
  auto protoInitializedDimensionsMap = *protoMetaData->mutable_fieldidtolegaldimensions();
  for(auto IDToLegalDimension : metaData.fieldIDToInitializedDimensionsMap_) {
    proto::iir::Array3i array;
    array.set_int1(IDToLegalDimension.second[0]);
    array.set_int2(IDToLegalDimension.second[1]);
    array.set_int3(IDToLegalDimension.second[2]);
    protoInitializedDimensionsMap.insert({IDToLegalDimension.first, array});
  }
  // Filling Field: map<string, GlobalValueAndType> GlobalVariableToValue = 14;
  auto protoGlobalVariableMap = *protoMetaData->mutable_globalvariabletovalue();
  for(auto& globalToValue : metaData.globalVariableMap_) {
    proto::iir::GlobalValueAndType protoGlobalToStore;
    int typekind = -1;
    double value;
    bool valueIsSet = false;
    switch(globalToValue.second->getType()) {
    case sir::Value::Boolean:
      if(!globalToValue.second->empty()) {
        value = globalToValue.second->getValue<bool>();
        valueIsSet = true;
      }
      typekind = 1;
      break;
    case sir::Value::Integer:
      if(!globalToValue.second->empty()) {
        value = globalToValue.second->getValue<int>();
        valueIsSet = true;
      }
      typekind = 2;
      break;
    case sir::Value::Double:
      if(!globalToValue.second->empty()) {
        value = globalToValue.second->getValue<double>();
        valueIsSet = true;
      }
      typekind = 3;
      break;
    default:
      dawn_unreachable("non-supported global type");
    }
    protoGlobalToStore.set_typekind(typekind);
    if(valueIsSet) {
      protoGlobalToStore.set_value(value);
    }
    protoGlobalToStore.set_valueisset(valueIsSet);
    protoGlobalVariableMap.insert({globalToValue.first, protoGlobalToStore});
  }
  // Filling Field: dawn.proto.statements.SourceLocation stencilLocation = 15;
  auto protoStencilLoc = protoMetaData->mutable_stencillocation();
  protoStencilLoc->set_column(metaData.stencilLocation_.Column);
  protoStencilLoc->set_line(metaData.stencilLocation_.Line);
  // Filling Field: string stencilMName = 16;
  protoMetaData->set_stencilname(metaData.stencilName_);
  // Filling Field: string fileName = 17;
  protoMetaData->set_filename(metaData.fileName_);
}

void IIRSerializer::serializeIIR(proto::iir::StencilInstantiation& target,
                                 const std::unique_ptr<iir::IIR>& iir) {
  auto protoIIR = target.mutable_internalir();
  // Get all the stencils
  for(const auto& stencils : iir->getChildren()) {
    // creation of a new protobuf stencil
    auto protoStencil = protoIIR->add_stencils();
    // Information other than the children
    protoStencil->set_stencilid(stencils->getStencilID());
    auto protoAttribute = protoStencil->mutable_attr();
    protoAttribute->set_attrbits(stencils->getStencilAttributes().getBits());

    // adding it's children
    for(const auto& multistages : stencils->getChildren()) {
      // creation of a protobuf multistage
      auto protoMSS = protoStencil->add_multistages();
      // Information other than the children
      if(multistages->getLoopOrder() == dawn::iir::LoopOrderKind::LK_Forward) {
        protoMSS->set_looporder(proto::iir::MultiStage::Forward);
      } else if(multistages->getLoopOrder() == dawn::iir::LoopOrderKind::LK_Backward) {
        protoMSS->set_looporder(proto::iir::MultiStage::Backward);
      } else {
        protoMSS->set_looporder(proto::iir::MultiStage::Parallel);
      }
      protoMSS->set_mulitstageid(multistages->getID());

      // adding it's children
      for(const auto& stages : multistages->getChildren()) {
        auto protoStage = protoMSS->add_stages();
        // Information other than the children
        protoStage->set_stageid(stages->getStageID());

        // adding it's children
        for(const auto& domethod : stages->getChildren()) {
          auto protoDoMethod = protoStage->add_domethods();
          // Information other than the children
          dawn::sir::Interval interval = domethod->getInterval().asSIRInterval();
          setInterval(protoDoMethod->mutable_interval(), &interval);
          protoDoMethod->set_domethodid(domethod->getID());

          // adding it's children
          for(const auto& stmtaccesspair : domethod->getChildren()) {
            auto protoStmtAccessPair = protoDoMethod->add_stmtaccesspairs();
            serializeStmtAccessPair(protoStmtAccessPair, stmtaccesspair);
            //            std::cout << "serializing this stmt\n"
            //                      <<
            //                      stmtaccesspair->toString(&stencils->getStencilInstantiation())
            //                      << std::endl;
          }
        }
      }
    }
  }
}

std::string
IIRSerializer::serializeImpl(const std::shared_ptr<iir::StencilInstantiation>& instantiation,
                             dawn::IIRSerializer::SerializationKind kind) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  using namespace dawn::proto::iir;
  proto::iir::StencilInstantiation protoStencilInstantiation;
  serializeMetaData(protoStencilInstantiation, instantiation->getMetaData());
  serializeIIR(protoStencilInstantiation, instantiation->getIIR());

  // Encode the message
  std::string str;
  switch(kind) {
  case dawn::IIRSerializer::SK_Json: {
    google::protobuf::util::JsonPrintOptions options;
    options.add_whitespace = true;
    options.always_print_primitive_fields = true;
    options.preserve_proto_field_names = true;
    auto status =
        google::protobuf::util::MessageToJsonString(protoStencilInstantiation, &str, options);
    if(!status.ok())
      throw std::runtime_error(dawn::format("cannot serialize IIR: %s", status.ToString()));
    break;
  }
  case dawn::IIRSerializer::SK_Byte: {
    if(!protoStencilInstantiation.SerializeToString(&str))
      throw std::runtime_error(dawn::format("cannot serialize IIR:"));
    break;
  }
  default:
    dawn_unreachable("invalid SerializationKind");
  }

  return str;
}

void IIRSerializer::deserializeMetaData(std::shared_ptr<iir::StencilInstantiation>& target,
                                        const proto::iir::StencilMetaInfo& protoMetaData) {
  auto& metadata = target->getMetaData();
  for(auto IDtoName : protoMetaData.accessidtoname()) {
    metadata.AccessIDToNameMap_.insert({IDtoName.first, IDtoName.second});
  }
  for(auto exprToID : protoMetaData.exprtoaccessid()) {
    metadata.ExprToAccessIDMap_[makeExpr(exprToID.expr())] = exprToID.ids();
  }
  for(auto stmtToID : protoMetaData.stmttoaccessid()) {
    metadata.StmtToAccessIDMap_[makeStmt(stmtToID.stmt())] = stmtToID.ids();
  }
  for(auto literalIDToName : protoMetaData.literalidtoname()) {
    metadata.LiteralAccessIDToNameMap_[literalIDToName.first] = literalIDToName.second;
  }
  for(int i = 0; i < protoMetaData.fieldaccessids_size(); ++i) {
    metadata.FieldAccessIDSet_.insert(protoMetaData.fieldaccessids(i));
  }
  for(int i = 0; i < protoMetaData.apifieldids_size(); ++i) {
    metadata.apiFieldIDs_.push_back(protoMetaData.apifieldids(i));
  }
  for(int i = 0; i < protoMetaData.temporaryfieldids_size(); ++i) {
    metadata.TemporaryFieldAccessIDSet_.insert(protoMetaData.temporaryfieldids(i));
  }
  for(int i = 0; i < protoMetaData.globalvariableids_size(); ++i) {
    metadata.GlobalVariableAccessIDSet_.insert(protoMetaData.globalvariableids(i));
  }
  //
  // Variable Versions
  //
  for(auto stencilDescStmt : protoMetaData.stencildescstatements()) {
    metadata.stencilDescStatements_.push_back(makeStatement(&stencilDescStmt));
  }
//  for(auto IDToCall : protoMetaData.idtostencilcall()) {
//    metadata.IDToStencilCallMap_[IDToCall.first] = makeStmt((IDToCall.second));
//  }
//  for(auto FieldnameToBC : protoMetaData.fieldnametoboundarycondition()) {
//    metadata.FieldnameToBoundaryConditionMap_[FieldnameToBC.first] = makeStmt((FieldnameToBC.second));
//  }
  for(auto fieldIDInitializedDims : protoMetaData.fieldidtolegaldimensions()) {
    Array3i dims{fieldIDInitializedDims.second.int1(), fieldIDInitializedDims.second.int2(),
                 fieldIDInitializedDims.second.int3()};
    metadata.fieldIDToInitializedDimensionsMap_[fieldIDInitializedDims.first] = dims;
  }
}

void IIRSerializer::deserializeIIR(std::shared_ptr<iir::StencilInstantiation>& target,
                                   const proto::iir::IIR& protoIIR) {
  for(const auto& protoStencils : protoIIR.stencils()) {
    std::cout << "And now we deserialize the stencil" << std::endl;
    sir::Attr attributes;
    attributes.setBits(protoStencils.attr().attrbits());
    //    target->getIIR()->insertChild(
    //        make_unique<iir::Stencil>(*target, attributes, protoStencils.stencilid()));

    for(const auto& protoMSS : protoStencils.multistages()) {
      std::cout << "And now we deserialize the mss" << std::endl;
      for(const auto& protoStage : protoMSS.stages()) {
        std::cout << "And now we deserialize the stage" << std::endl;
        for(const auto& protoDoMethod : protoStage.domethods()) {
          std::cout << "And now we deserialize the domethod" << std::endl;
          for(const auto& protoStmtAccessPair : protoDoMethod.stmtaccesspairs()) {
            std::cout << "And now we deserialize the stmtaccesspair" << std::endl;
            auto stmt = makeStmt(protoStmtAccessPair.statement().aststmt());
            std::cout << stmt << std::endl;
          }
        }
      }
    }
  }
}

void IIRSerializer::deserializeImpl(const std::string& str, IIRSerializer::SerializationKind kind,
                                    std::shared_ptr<iir::StencilInstantiation>& target) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  // Decode the string
  proto::iir::StencilInstantiation protoStencilInstantiation;
  switch(kind) {
  case dawn::IIRSerializer::SK_Json: {
    auto status = google::protobuf::util::JsonStringToMessage(str, &protoStencilInstantiation);
    if(!status.ok())
      throw std::runtime_error(
          dawn::format("cannot deserialize StencilInstantiation: %s", status.ToString()));
    break;
  }
  case dawn::IIRSerializer::SK_Byte: {
    if(!protoStencilInstantiation.ParseFromString(str))
      throw std::runtime_error(dawn::format("cannot deserialize StencilInstantiation: %s")); //,
    // ProtobufLogger::getInstance().getErrorMessagesAndReset()));
    break;
  }
  default:
    dawn_unreachable("invalid SerializationKind");
  }

  std::shared_ptr<iir::StencilInstantiation> instantiation =
      std::make_shared<iir::StencilInstantiation>(target->getOptimizerContext());

  deserializeMetaData(instantiation, (protoStencilInstantiation.metadata()));
  deserializeIIR(instantiation, (protoStencilInstantiation.internalir()));

  target = instantiation;
}

void dawn::IIRSerializer::deserialize(const std::string& file,
                                      std::shared_ptr<iir::StencilInstantiation> instantiation,
                                      dawn::IIRSerializer::SerializationKind kind) {
  std::ifstream ifs(file);
  if(!ifs.is_open())
    throw std::runtime_error(
        dawn::format("cannot deserialize IIR: failed to open file \"%s\"", file));

  std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  deserializeImpl(str, kind, instantiation);
}

void dawn::IIRSerializer::deserializeFromString(
    const std::string& str, std::shared_ptr<iir::StencilInstantiation> instantiation,
    dawn::IIRSerializer::SerializationKind kind) {
  deserializeImpl(str, kind, instantiation);
}

void dawn::IIRSerializer::serialize(const std::string& file,
                                    const std::shared_ptr<iir::StencilInstantiation> instantiation,
                                    dawn::IIRSerializer::SerializationKind kind) {
  std::ofstream ofs(file);
  if(!ofs.is_open())
    throw std::runtime_error(format("cannot serialize SIR: failed to open file \"%s\"", file));

  auto str = serializeImpl(instantiation, kind);
  std::copy(str.begin(), str.end(), std::ostreambuf_iterator<char>(ofs));
}

std::string dawn::IIRSerializer::serializeToString(
    const std::shared_ptr<iir::StencilInstantiation> instantiation,
    dawn::IIRSerializer::SerializationKind kind) {
  return serializeImpl(instantiation, kind);
}
