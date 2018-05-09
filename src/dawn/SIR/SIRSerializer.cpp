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

#include "dawn/SIR/SIR.h"
#include "dawn/SIR/AST.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.pb.h"
#include "dawn/SIR/SIRSerializer.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/Unreachable.h"
#include <fstream>
#include <google/protobuf/util/json_util.h>
#include <list>
#include <stack>
#include <tuple>

namespace dawn {

namespace {

/// @brief Singleton logger of Protobuf
class ProtobufLogger : public NonCopyable {
public:
  using LogMessage = std::tuple<google::protobuf::LogLevel, std::string, int, std::string>;

  /// @brief Protobufs internal logging handler
  static void LogHandler(google::protobuf::LogLevel level, const char* filename, int line,
                         const std::string& message) {
    // Log to the Dawn logger
    switch(level) {
    case google::protobuf::LOGLEVEL_INFO:
      DAWN_LOG(INFO) << "Protobuf: " << message;
      break;
    case google::protobuf::LOGLEVEL_WARNING:
      DAWN_LOG(WARNING) << "Protobuf: " << message;
      break;
    case google::protobuf::LOGLEVEL_ERROR:
      DAWN_LOG(ERROR) << "Protobuf: " << message;
      break;
    case google::protobuf::LOGLEVEL_FATAL:
      DAWN_LOG(FATAL) << "Protobuf: " << message;
      break;
    }

    // Cache the messages
    getInstance().push(LogMessage{level, filename, line, message});
  }

  /// @brief Push a `message` to the logging stack
  void push(LogMessage message) { logStack_.emplace_back(std::move(message)); }

  /// @brief Get a dump of all error messages (in the order of occurence) and reset the internal
  /// logging stack
  std::string getErrorMessagesAndReset() {
    std::string str = "Protobuf errors (most recent call last):\n\n";
    for(const LogMessage& msg : logStack_)
      if(std::get<0>(msg) >= google::protobuf::LOGLEVEL_ERROR)
        str += dawn::format("%s:%i: %s\n\n");
    logStack_.clear();
    return str;
  }

  /// @brief Initialize and register the Logger
  static void init() {
    if(instance_)
      return;
    instance_ = new ProtobufLogger();
    google::protobuf::SetLogHandler(ProtobufLogger::LogHandler);
  }

  /// @brief Get the singleton instance of the logger
  static ProtobufLogger& getInstance() noexcept { return *instance_; }

private:
  std::list<LogMessage> logStack_;

  static ProtobufLogger* instance_;
};

ProtobufLogger* ProtobufLogger::instance_ = nullptr;

//===------------------------------------------------------------------------------------------===//
//     Serialization
//===------------------------------------------------------------------------------------------===//

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
  for(const auto& initializedDimension : field->fieldDimensions) {
    fieldProto->add_field_dimensions(initializedDimension);
  }
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

    sir::VerticalRegion* verticalRegion = stmt->getVerticalRegion().get();
    sir::proto::VerticalRegion* verticalRegionProto = protoStmt->mutable_vertical_region();

    // VerticalRegion.Loc
    setLocation(verticalRegionProto->mutable_loc(), verticalRegion->Loc);

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

static void setAST(sir::proto::AST* astProto, const AST* ast) {
  ProtoStmtBuilder builder(astProto->mutable_root());
  ast->accept(builder);
}

static std::string serializeImpl(const SIR* sir, SIRSerializer::SerializationKind kind) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  ProtobufLogger::init();

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
    setLocation(stencilProto->mutable_loc(), stencil->Loc);

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
    setLocation(stencilFunctionProto->mutable_loc(), stencilFunction->Loc);

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
    if(!value.empty()) {
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
    }

    mapProto->insert({name, valueProto});
  }

  // Encode the message
  std::string str;
  switch(kind) {
  case dawn::SIRSerializer::SK_Json: {
    google::protobuf::util::JsonPrintOptions options;
    options.add_whitespace = true;
    options.always_print_primitive_fields = true;
    options.preserve_proto_field_names = true;
    auto status = google::protobuf::util::MessageToJsonString(sirProto, &str, options);
    if(!status.ok())
      throw std::runtime_error(format("cannot serialize SIR: %s", status.ToString()));
    break;
  }
  case dawn::SIRSerializer::SK_Byte: {
    if(!sirProto.SerializeToString(&str))
      throw std::runtime_error(dawn::format(
          "cannot deserialize SIR: %s", ProtobufLogger::getInstance().getErrorMessagesAndReset()));
    break;
  }
  default:
    dawn_unreachable("invalid SerializationKind");
  }

  return str;
}

} // anonymous namespace

void SIRSerializer::serialize(const std::string& file, const SIR* sir, SerializationKind kind) {
  std::ofstream ofs(file);
  if(!ofs.is_open())
    throw std::runtime_error(format("cannot serialize SIR: failed to open file \"%s\"", file));

  auto str = serializeImpl(sir, kind);
  std::copy(str.begin(), str.end(), std::ostreambuf_iterator<char>(ofs));
}

std::string SIRSerializer::serializeToString(const SIR* sir, SerializationKind kind) {
  return serializeImpl(sir, kind);
}

//===------------------------------------------------------------------------------------------===//
//     Deserialization
//===------------------------------------------------------------------------------------------===//

namespace {

static std::shared_ptr<AST> makeAST(const sir::proto::AST& astProto);

template <class T>
static SourceLocation makeLocation(const T& proto) {
  return proto.has_loc() ? SourceLocation(proto.loc().line(), proto.loc().column())
                         : SourceLocation{};
}

static std::shared_ptr<sir::Field> makeField(const sir::proto::Field& fieldProto) {
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

static BuiltinTypeID makeBuiltinTypeID(const sir::proto::BuiltinType& builtinTypeProto) {
  switch(builtinTypeProto.type_id()) {
  case sir::proto::BuiltinType_TypeID_Invalid:
    return BuiltinTypeID::Invalid;
  case sir::proto::BuiltinType_TypeID_Auto:
    return BuiltinTypeID::Auto;
  case sir::proto::BuiltinType_TypeID_Boolean:
    return BuiltinTypeID::Boolean;
  case sir::proto::BuiltinType_TypeID_Integer:
    return BuiltinTypeID::Integer;
  case sir::proto::BuiltinType_TypeID_Float:
    return BuiltinTypeID::Float;
  default:
    return BuiltinTypeID::Invalid;
  }
  return BuiltinTypeID::Invalid;
}

static std::shared_ptr<sir::Direction> makeDirection(const sir::proto::Direction& directionProto) {
  return std::make_shared<sir::Direction>(directionProto.name(), makeLocation(directionProto));
}

static std::shared_ptr<sir::Offset> makeOffset(const sir::proto::Offset& offsetProto) {
  return std::make_shared<sir::Offset>(offsetProto.name(), makeLocation(offsetProto));
}

static std::shared_ptr<sir::Interval> makeInterval(const sir::proto::Interval& intervalProto) {
  int lowerLevel = -1, upperLevel = -1, lowerOffset = -1, upperOffset = -1;

  if(intervalProto.LowerLevel_case() == sir::proto::Interval::kSpecialLowerLevel)
    lowerLevel = intervalProto.special_lower_level() ==
                         sir::proto::Interval_SpecialLevel::Interval_SpecialLevel_Start
                     ? sir::Interval::Start
                     : sir::Interval::End;
  else
    lowerLevel = intervalProto.lower_level();

  if(intervalProto.UpperLevel_case() == sir::proto::Interval::kSpecialUpperLevel)
    upperLevel = intervalProto.special_upper_level() ==
                         sir::proto::Interval_SpecialLevel::Interval_SpecialLevel_Start
                     ? sir::Interval::Start
                     : sir::Interval::End;
  else
    upperLevel = intervalProto.upper_level();

  lowerOffset = intervalProto.lower_offset();
  upperOffset = intervalProto.upper_offset();
  return std::make_shared<sir::Interval>(lowerLevel, upperLevel, lowerOffset, upperOffset);
}

static std::shared_ptr<sir::VerticalRegion>
makeVerticalRegion(const sir::proto::VerticalRegion& verticalRegionProto) {
  // VerticalRegion.Loc
  auto loc = makeLocation(verticalRegionProto);

  // VerticalRegion.Ast
  auto ast = makeAST(verticalRegionProto.ast());

  // VerticalRegion.VerticalInterval
  auto interval = makeInterval(verticalRegionProto.interval());

  // VerticalRegion.LoopOrder
  auto loopOrder = verticalRegionProto.loop_order() == sir::proto::VerticalRegion::Backward
                       ? sir::VerticalRegion::LK_Backward
                       : sir::VerticalRegion::LK_Forward;

  return std::make_shared<sir::VerticalRegion>(ast, interval, loopOrder, loc);
}

static std::shared_ptr<sir::StencilCall>
makeStencilCall(const sir::proto::StencilCall& stencilCallProto) {
  auto stencilCall =
      std::make_shared<sir::StencilCall>(stencilCallProto.callee(), makeLocation(stencilCallProto));

  for(const auto& arg : stencilCallProto.arguments())
    stencilCall->Args.emplace_back(makeField(arg));

  return stencilCall;
}

static std::shared_ptr<Expr> makeExpr(const sir::proto::Expr& expressionProto) {
  switch(expressionProto.expr_case()) {
  case sir::proto::Expr::kUnaryOperator: {
    const auto& exprProto = expressionProto.unary_operator();
    return std::make_shared<UnaryOperator>(makeExpr(exprProto.operand()), exprProto.op(),
                                           makeLocation(exprProto));
  }
  case sir::proto::Expr::kBinaryOperator: {
    const auto& exprProto = expressionProto.binary_operator();
    return std::make_shared<BinaryOperator>(makeExpr(exprProto.left()), exprProto.op(),
                                            makeExpr(exprProto.right()), makeLocation(exprProto));
  }
  case sir::proto::Expr::kAssignmentExpr: {
    const auto& exprProto = expressionProto.assignment_expr();
    return std::make_shared<AssignmentExpr>(makeExpr(exprProto.left()), makeExpr(exprProto.right()),
                                            exprProto.op(), makeLocation(exprProto));
  }
  case sir::proto::Expr::kTernaryOperator: {
    const auto& exprProto = expressionProto.ternary_operator();
    return std::make_shared<TernaryOperator>(makeExpr(exprProto.cond()), makeExpr(exprProto.left()),
                                             makeExpr(exprProto.right()), makeLocation(exprProto));
  }
  case sir::proto::Expr::kFunCallExpr: {
    const auto& exprProto = expressionProto.fun_call_expr();
    auto expr = std::make_shared<FunCallExpr>(exprProto.callee(), makeLocation(exprProto));
    for(const auto& argProto : exprProto.arguments())
      expr->getArguments().emplace_back(makeExpr(argProto));
    return expr;
  }
  case sir::proto::Expr::kStencilFunCallExpr: {
    const auto& exprProto = expressionProto.stencil_fun_call_expr();
    auto expr = std::make_shared<StencilFunCallExpr>(exprProto.callee(), makeLocation(exprProto));
    for(const auto& argProto : exprProto.arguments())
      expr->getArguments().emplace_back(makeExpr(argProto));
    return expr;
  }
  case sir::proto::Expr::kStencilFunArgExpr: {
    const auto& exprProto = expressionProto.stencil_fun_arg_expr();
    int direction = -1, offset = 0, argumentIndex = -1; // default values

    if(exprProto.has_dimension()) {
      switch(exprProto.dimension().direction()) {
      case sir::proto::Dimension_Direction_I:
        direction = 0;
        break;
      case sir::proto::Dimension_Direction_J:
        direction = 1;
        break;
      case sir::proto::Dimension_Direction_K:
        direction = 2;
        break;
      case sir::proto::Dimension_Direction_Invalid:
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
  case sir::proto::Expr::kVarAccessExpr: {
    const auto& exprProto = expressionProto.var_access_expr();
    auto expr = std::make_shared<VarAccessExpr>(
        exprProto.name(), exprProto.has_index() ? makeExpr(exprProto.index()) : nullptr,
        makeLocation(exprProto));
    expr->setIsExternal(exprProto.is_external());
    return expr;
  }
  case sir::proto::Expr::kFieldAccessExpr: {
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
  case sir::proto::Expr::kLiteralAccessExpr: {
    const auto& exprProto = expressionProto.literal_access_expr();
    return std::make_shared<LiteralAccessExpr>(
        exprProto.value(), makeBuiltinTypeID(exprProto.type()), makeLocation(exprProto));
  }
  case sir::proto::Expr::EXPR_NOT_SET:
  default:
    dawn_unreachable("expr not set");
  }
  return nullptr;
}

static std::shared_ptr<Stmt> makeStmt(const sir::proto::Stmt& statementProto) {
  switch(statementProto.stmt_case()) {
  case sir::proto::Stmt::kBlockStmt: {
    const auto& stmtProto = statementProto.block_stmt();
    auto stmt = std::make_shared<BlockStmt>(makeLocation(stmtProto));

    for(const auto& s : stmtProto.statements())
      stmt->push_back(makeStmt(s));

    return stmt;
  }
  case sir::proto::Stmt::kExprStmt: {
    const auto& stmtProto = statementProto.expr_stmt();
    return std::make_shared<ExprStmt>(makeExpr(stmtProto.expr()), makeLocation(stmtProto));
  }
  case sir::proto::Stmt::kReturnStmt: {
    const auto& stmtProto = statementProto.return_stmt();
    return std::make_shared<ReturnStmt>(makeExpr(stmtProto.expr()), makeLocation(stmtProto));
  }
  case sir::proto::Stmt::kVarDeclStmt: {
    const auto& stmtProto = statementProto.var_decl_stmt();

    std::vector<std::shared_ptr<Expr>> initList;
    for(const auto& e : stmtProto.init_list())
      initList.emplace_back(makeExpr(e));

    const sir::proto::Type& typeProto = stmtProto.type();
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
  case sir::proto::Stmt::kStencilCallDeclStmt: {
    const auto& stmtProto = statementProto.stencil_call_decl_stmt();
    return std::make_shared<StencilCallDeclStmt>(makeStencilCall(stmtProto.stencil_call()),
                                                 makeLocation(stmtProto));
  }
  case sir::proto::Stmt::kVerticalRegionDeclStmt: {
    const auto& stmtProto = statementProto.vertical_region_decl_stmt();
    return std::make_shared<VerticalRegionDeclStmt>(makeVerticalRegion(stmtProto.vertical_region()),
                                                    makeLocation(stmtProto));
  }
  case sir::proto::Stmt::kBoundaryConditionDeclStmt: {
    const auto& stmtProto = statementProto.boundary_condition_decl_stmt();
    auto stmt =
        std::make_shared<BoundaryConditionDeclStmt>(stmtProto.functor(), makeLocation(stmtProto));
    for(const auto& fieldProto : stmtProto.fields())
      stmt->getFields().emplace_back(makeField(fieldProto));
    return stmt;
  }
  case sir::proto::Stmt::kIfStmt: {
    const auto& stmtProto = statementProto.if_stmt();
    return std::make_shared<IfStmt>(
        makeStmt(stmtProto.cond_part()), makeStmt(stmtProto.then_part()),
        stmtProto.has_else_part() ? makeStmt(stmtProto.else_part()) : nullptr,
        makeLocation(stmtProto));
  }
  case sir::proto::Stmt::STMT_NOT_SET:
  default:
    dawn_unreachable("stmt not set");
  }
  return nullptr;
}

static std::shared_ptr<AST> makeAST(const sir::proto::AST& astProto) {
  auto ast = std::make_shared<AST>();
  auto root = dyn_pointer_cast<BlockStmt>(makeStmt(astProto.root()));
  if(!root)
    throw std::runtime_error("root statement of AST is not a 'BlockStmt'");
  ast->setRoot(root);
  return ast;
}

static std::shared_ptr<SIR> deserializeImpl(const std::string& str,
                                            SIRSerializer::SerializationKind kind) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  using namespace sir;
  ProtobufLogger::init();

  // Decode the string
  sir::proto::SIR sirProto;
  switch(kind) {
  case dawn::SIRSerializer::SK_Json: {
    auto status = google::protobuf::util::JsonStringToMessage(str, &sirProto);
    if(!status.ok())
      throw std::runtime_error(dawn::format("cannot deserialize SIR: %s", status.ToString()));
    break;
  }
  case dawn::SIRSerializer::SK_Byte: {
    if(!sirProto.ParseFromString(str))
      throw std::runtime_error(dawn::format(
          "cannot deserialize SIR: %s", ProtobufLogger::getInstance().getErrorMessagesAndReset()));
    break;
  }
  default:
    dawn_unreachable("invalid SerializationKind");
  }

  // Convert protobuf SIR to SIR
  std::shared_ptr<SIR> sir = std::make_shared<SIR>();

  try {
    // SIR.Filename
    sir->Filename = sirProto.filename();

    // SIR.Stencils
    for(const sir::proto::Stencil& stencilProto : sirProto.stencils()) {
      std::shared_ptr<Stencil> stencil = std::make_shared<Stencil>();

      // Stencil.Name
      stencil->Name = stencilProto.name();

      // Stencil.Loc
      stencil->Loc = makeLocation(stencilProto);

      // Stencil.StencilDescAst
      stencil->StencilDescAst = makeAST(stencilProto.ast());

      // Stencil.Fields
      for(const sir::proto::Field& fieldProto : stencilProto.fields())
        stencil->Fields.emplace_back(makeField(fieldProto));

      sir->Stencils.emplace_back(stencil);
    }

    // SIR.StencilFunctions
    for(const sir::proto::StencilFunction& stencilFunctionProto : sirProto.stencil_functions()) {
      std::shared_ptr<StencilFunction> stencilFunction = std::make_shared<StencilFunction>();

      // StencilFunction.Name
      stencilFunction->Name = stencilFunctionProto.name();

      // Stencil.Loc
      stencilFunction->Loc = makeLocation(stencilFunctionProto);

      // StencilFunction.Args
      for(const sir::proto::StencilFunctionArg& sirArg : stencilFunctionProto.arguments()) {
        switch(sirArg.Arg_case()) {
        case sir::proto::StencilFunctionArg::kFieldValue:
          stencilFunction->Args.emplace_back(makeField(sirArg.field_value()));
          break;
        case sir::proto::StencilFunctionArg::kDirectionValue:
          stencilFunction->Args.emplace_back(makeDirection(sirArg.direction_value()));
          break;
        case sir::proto::StencilFunctionArg::kOffsetValue:
          stencilFunction->Args.emplace_back(makeOffset(sirArg.offset_value()));
          break;
        case sir::proto::StencilFunctionArg::ARG_NOT_SET:
        default:
          dawn_unreachable("argument not set");
        }
      }

      // StencilFunction.Intervals
      for(const sir::proto::Interval& sirInterval : stencilFunctionProto.intervals())
        stencilFunction->Intervals.emplace_back(makeInterval(sirInterval));

      // StencilFunction.Asts
      for(const sir::proto::AST& sirAst : stencilFunctionProto.asts())
        stencilFunction->Asts.emplace_back(makeAST(sirAst));

      sir->StencilFunctions.emplace_back(stencilFunction);
    }

    // SIR.GlobalVariableMap
    for(const auto& nameValuePair : sirProto.global_variables().map()) {
      const std::string& sirName = nameValuePair.first;
      const sir::proto::GlobalVariableValue& sirValue = nameValuePair.second;
      std::shared_ptr<Value> value = nullptr;

      switch(sirValue.Value_case()) {
      case sir::proto::GlobalVariableValue::kBooleanValue:
        value = std::make_shared<Value>(static_cast<bool>(sirValue.boolean_value()));
        break;
      case sir::proto::GlobalVariableValue::kIntegerValue:
        value = std::make_shared<Value>(static_cast<int>(sirValue.integer_value()));
        break;
      case sir::proto::GlobalVariableValue::kDoubleValue:
        value = std::make_shared<Value>(static_cast<double>(sirValue.double_value()));
        break;
      case sir::proto::GlobalVariableValue::kStringValue:
        value = std::make_shared<Value>(static_cast<std::string>(sirValue.string_value()));
        break;
      case sir::proto::GlobalVariableValue::VALUE_NOT_SET:
      default:
        dawn_unreachable("value not set");
      }

      value->setIsConstexpr(sirValue.is_constexpr());
      sir->GlobalVariableMap->emplace(sirName, std::move(value));
    }

  } catch(std::runtime_error& error) {
    throw std::runtime_error(dawn::format("cannot deserialize SIR: %s", error.what()));
  }
  return sir;
}

} // anonymous namespace

std::shared_ptr<SIR> SIRSerializer::deserialize(const std::string& file, SerializationKind kind) {
  std::ifstream ifs(file);
  if(!ifs.is_open())
    throw std::runtime_error(
        dawn::format("cannot deserialize SIR: failed to open file \"%s\"", file));

  std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  return deserializeImpl(str, kind);
}

std::shared_ptr<SIR> SIRSerializer::deserializeFromString(const std::string& str,
                                                          SerializationKind kind) {
  return deserializeImpl(str, kind);
}

} // namespace dawn
