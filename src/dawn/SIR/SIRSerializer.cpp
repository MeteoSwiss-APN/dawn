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
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include "dawn/SIR/SIR.pb.h"
#include "dawn/SIR/SIRSerializer.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/Unreachable.h"
#include <fstream>
#include <google/protobuf/util/json_util.h>
#include <stack>

namespace dawn {

namespace {

static void setAST(sir::proto::AST* astProto, const AST* ast);

static void setLocation(sir::proto::SourceLocation* locProto, const SourceLocation& loc) {
  locProto->set_column(loc.Column);
  locProto->set_line(loc.Line);
}

static void setBuiltinType(sir::proto::BuiltinType* builtinTypeProto,
                           const BuiltinTypeID& builtinType) {
  builtinTypeProto->set_type_id(static_cast<sir::proto::BuiltinType_TypeID>(builtinType));
}

static void setInterval(sir::proto::Interval* intervalProto, const sir::Interval* interval) {
  if(interval->LowerLevel == sir::Interval::Start)
    intervalProto->set_special_lower_level(sir::proto::Interval::Start);
  else if(interval->LowerLevel == sir::Interval::End)
    intervalProto->set_special_lower_level(sir::proto::Interval::End);
  else
    intervalProto->set_lower_level(interval->LowerLevel);

  if(interval->UpperLevel == sir::Interval::Start)
    intervalProto->set_special_upper_level(sir::proto::Interval::Start);
  else if(interval->UpperLevel == sir::Interval::End)
    intervalProto->set_special_upper_level(sir::proto::Interval::End);
  else
    intervalProto->set_upper_level(interval->UpperLevel);

  intervalProto->set_lower_offset(interval->LowerOffset);
  intervalProto->set_upper_offset(interval->UpperOffset);
}

static void setField(sir::proto::Field* fieldProto, const sir::Field* field) {
  fieldProto->set_name(field->Name);
  fieldProto->set_is_temporary(field->IsTemporary);
  setLocation(fieldProto->mutable_loc(), field->Loc);
}

static void setDirection(sir::proto::Direction* directionProto, const sir::Direction* direction) {
  directionProto->set_name(direction->Name);
  setLocation(directionProto->mutable_loc(), direction->Loc);
}

static void setOffset(sir::proto::Offset* offsetProto, const sir::Offset* offset) {
  offsetProto->set_name(offset->Name);
  setLocation(offsetProto->mutable_loc(), offset->Loc);
}

class ProtoStmtBuilder : public ASTVisitor {
  std::stack<sir::proto::Stmt*> currentStmtProto_;
  std::stack<sir::proto::Expr*> currentExprProto_;

public:
  ProtoStmtBuilder(sir::proto::Stmt* stmtProto) { currentStmtProto_.push(stmtProto); }

  sir::proto::Stmt* getCurrentStmtProto() {
    DAWN_ASSERT(!currentStmtProto_.empty());
    return currentStmtProto_.top();
  }

  sir::proto::Expr* getCurrentExprProto() {
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
      setBuiltinType(protoStmt->mutable_type()->mutable_builtin_type(), stmt->getType().getBuiltinTypeID());
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

    sir::VerticalRegion* verticalRegion = stmt->getVerticalRegion().get();
    sir::proto::VerticalRegion* verticalRegionProto = protoStmt->mutable_vertical_region();

    // VerticalRegion.Loc
    verticalRegionProto->mutable_loc()->set_column(verticalRegion->Loc.Column);
    verticalRegionProto->mutable_loc()->set_line(verticalRegion->Loc.Line);

    // VerticalRegion.Ast
    setAST(verticalRegionProto->mutable_ast(), verticalRegion->Ast.get());

    // VerticalRegion.VerticalInterval
    setInterval(verticalRegionProto->mutable_interval(), verticalRegion->VerticalInterval.get());

    // VerticalRegion.LoopOrder
    verticalRegionProto->set_loop_order(verticalRegion->LoopOrder ==
                                                sir::VerticalRegion::LK_Backward
                                            ? sir::proto::VerticalRegion::Backward
                                            : sir::proto::VerticalRegion::Forward);
    
    setLocation(protoStmt->mutable_loc(), stmt->getSourceLocation());        
  }
  
  void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) override {  
    auto protoStmt = getCurrentStmtProto()->mutable_stencil_call_decl_stmt();
  
    sir::StencilCall* stencilCall = stmt->getStencilCall().get();
    sir::proto::StencilCall* stencilCallProto = protoStmt->mutable_stencil_call();
  
    // StencilCall.Loc
    stencilCallProto->mutable_loc()->set_column(stencilCall->Loc.Column);
    stencilCallProto->mutable_loc()->set_line(stencilCall->Loc.Line);
  
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
    
    protoExpr->mutable_dimension()->set_dimension(
        expr->getDimension() == -1
            ? sir::proto::Dimension::Invalid
            : static_cast<sir::proto::Dimension_Direction>(expr->getDimension()));
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

static void setAST(sir::proto::AST* astProto, const AST* ast) {
  ProtoStmtBuilder builder(astProto->mutable_root());
  ast->accept(builder);
}

static std::string serializeImpl(const SIR* sir) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  // Convert SIR to protobuf SIR
  sir::proto::SIR sirProto;

  // SIR.Filename
  sirProto.set_filename(sir->Filename);

  // SIR.Stencils
  for(const auto& stencil : sir->Stencils) {
    auto stencilProto = sirProto.add_stencils();

    // Stencil.Name
    stencilProto->set_name(stencil->Name);

    // Stencil.Loc
    stencilProto->mutable_loc()->set_column(stencil->Loc.Column);
    stencilProto->mutable_loc()->set_line(stencil->Loc.Line);

    // Stencil.StencilDescAst
    setAST(stencilProto->mutable_ast(), stencil->StencilDescAst.get());

    // Stencil.Fields
    for(const auto& field : stencil->Fields) {
      auto fieldProto = stencilProto->add_fields();
      setField(fieldProto, field.get());
    }
  }
  
  // SIR.StencilFunctions
  for(const auto& stencilFunction : sir->StencilFunctions) {
    auto stencilFunctionProto = sirProto.add_stencil_functions();
  
    // StencilFunction.Name
    stencilFunctionProto->set_name(stencilFunction->Name);
  
    // StencilFunction.Loc
    stencilFunctionProto->mutable_loc()->set_column(stencilFunction->Loc.Column);
    stencilFunctionProto->mutable_loc()->set_line(stencilFunction->Loc.Line);
  
    // StencilFunction.Args
    for(const auto& arg : stencilFunction->Args) {
      auto argProto = stencilFunctionProto->add_arguments();
      if(sir::Field* field = dyn_cast<sir::Field>(arg.get())) {
        setField(argProto->mutable_field_value(), field);
      } else if(sir::Direction* direction = dyn_cast<sir::Direction>(arg.get())) {
        setDirection(argProto->mutable_direction_value(), direction);
      } else if(sir::Offset* offset = dyn_cast<sir::Offset>(arg.get())) {
        setOffset(argProto->mutable_offset_value(), offset);
      } else {
        dawn_unreachable("invalid argument");
      }
    }
  
    // StencilFunction.Intervals
    for(const auto& interval : stencilFunction->Intervals) {
      auto intervalProto = stencilFunctionProto->add_intervals();
      setInterval(intervalProto, interval.get());
    }
  
    // StencilFunction.Asts
    for(const auto& ast : stencilFunction->Asts) {
      auto astProto = stencilFunctionProto->add_asts();
      setAST(astProto, ast.get());
    }
  }
  
  // SIR.GlobalVariableMap  
  auto mapProto = sirProto.mutable_global_variables()->mutable_map();
  for(const auto& nameValuePair : *sir->GlobalVariableMap) { 
    const std::string& name = nameValuePair.first;
    const sir::Value& value = *nameValuePair.second;
    
    sir::proto::GlobalVariableValue valueProto;
    valueProto.set_is_constexpr(value.isConstexpr());
    switch(value.getType()) {
      case sir::Value::Boolean:
        valueProto.set_boolean_value(value.getValue<bool>());
        break;
      case sir::Value::Integer:
        valueProto.set_integer_value(value.getValue<int>());        
        break;
      case sir::Value::Double:
        valueProto.set_double_value(value.getValue<double>());        
        break;
      case sir::Value::String:
        valueProto.set_string_value(value.getValue<std::string>());        
        break;
      case sir::Value::None:
        break;
    }
    
    mapProto->insert({name, valueProto});
  }
  
  // Encode message to a  JSON formatted string
  std::string str;
  google::protobuf::util::JsonPrintOptions options;
  options.add_whitespace = true;
  options.always_print_primitive_fields = true;
  options.preserve_proto_field_names = true;
  auto status = google::protobuf::util::MessageToJsonString(sirProto, &str, options);
  if(!status.ok())
    throw std::runtime_error(dawn::format("cannot serialize SIR: %s", status.ToString()));
  return str;
}

static std::shared_ptr<SIR> deserializeImpl(const std::string& str) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  // Decode JSON formatted string
  sir::proto::SIR sirProto;
  auto status = google::protobuf::util::JsonStringToMessage(str, &sirProto);
  if(!status.ok())
    throw std::runtime_error(dawn::format("cannot deserialize SIR: %s", status.ToString()));

  // Convert protobuf SIR to SIR
  auto sir = std::make_shared<SIR>();
  return sir;
}

} // namespace internal

std::shared_ptr<SIR> SIRSerializer::deserialize(const std::string& file) {
  std::ifstream ifs(file);
  if(!ifs.is_open())
    throw std::runtime_error(
        dawn::format("cannot deserialize SIR: failed to open file \"%s\"", file));

  std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  return deserializeImpl(str);
}

std::shared_ptr<SIR> SIRSerializer::deserializeFromString(const std::string& str) {
  return deserializeImpl(str);
}

void SIRSerializer::serialize(const std::string& file, const SIR* sir) {
  std::ofstream ofs(file);
  if(!ofs.is_open())
    throw std::runtime_error(
        dawn::format("cannot serialize SIR: failed to open file \"%s\"", file));

  auto str = serializeImpl(sir);
  std::copy(str.begin(), str.end(), std::ostreambuf_iterator<char>(ofs));
}

std::string SIRSerializer::serializeToString(const SIR* sir) { return serializeImpl(sir); }

} // namespace dawn
