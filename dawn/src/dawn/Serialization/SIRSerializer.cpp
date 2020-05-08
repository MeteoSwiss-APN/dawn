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

#include "dawn/Serialization/SIRSerializer.h"
#include "dawn/SIR/AST.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include "dawn/SIR/SIR/SIR.pb.h"
#include "dawn/SIR/SIR/statements.pb.h"
#include "dawn/Serialization/ASTSerializer.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "dawn/Support/Exception.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/Logger.h"
#include "dawn/Support/Unreachable.h"
#include <fstream>
#include <google/protobuf/util/json_util.h>
#include <list>
#include <memory>
#include <stack>
#include <stdexcept>
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
      throw SyntacticError(std::string("[ERROR] Protobuf error: ") + message);
      break;
    case google::protobuf::LOGLEVEL_FATAL:
      throw SyntacticError(std::string("[FATAL] Protobuf error occurred: ") + message);
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

} // anonymous namespace

SIRSerializer::Format SIRSerializer::parseFormatString(const std::string& format) {
  if(format == "Byte" || format == "byte" || format == "BYTE")
    return SIRSerializer::Format::Byte;
  else if(format == "Json" || format == "json" || format == "JSON")
    return SIRSerializer::Format::Json;
  else
    throw std::invalid_argument(std::string("SIRSerializer::Format parse failed: ") + format);
}

//===------------------------------------------------------------------------------------------===//
//     Serialization
//===------------------------------------------------------------------------------------------===//

static std::string serializeImpl(const SIR* sir, SIRSerializer::Format kind) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  ProtobufLogger::init();

  // Convert SIR to protobuf SIR
  sir::proto::SIR sirProto;

  // SIR.GridType
  switch(sir->GridType) {
  case ast::GridType::Cartesian:
    sirProto.set_gridtype(proto::enums::GridType::Cartesian);
    break;
  case ast::GridType::Unstructured:
    sirProto.set_gridtype(proto::enums::GridType::Unstructured);
    break;
  }

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
        throw std::invalid_argument("Invalid argument: StencilFunction.Args");
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
  for(const auto& [name, value] : *sir->GlobalVariableMap) {

    // is_constexpr
    sir::proto::GlobalVariableValue valueProto;
    valueProto.set_is_constexpr(value.isConstexpr());

    // Value
    switch(value.getType()) {
    case sir::Value::Kind::Boolean:
      valueProto.set_boolean_value(value.has_value() ? value.getValue<bool>() : bool());
      break;
    case sir::Value::Kind::Integer:
      valueProto.set_integer_value(value.has_value() ? value.getValue<int>() : int());
      break;
    case sir::Value::Kind::Double:
      valueProto.set_double_value(value.has_value() ? value.getValue<double>() : double());
      break;
    case sir::Value::Kind::Float:
      valueProto.set_float_value(value.has_value() ? value.getValue<float>() : float());
      break;
    case sir::Value::Kind::String:
      valueProto.set_string_value(value.has_value() ? value.getValue<std::string>()
                                                    : std::string());
      break;
    }

    mapProto->insert({name, valueProto});
  }

  // Encode the message
  std::string str;
  switch(kind) {
  case dawn::SIRSerializer::Format::Json: {
    google::protobuf::util::JsonPrintOptions options;
    options.add_whitespace = true;
    options.always_print_primitive_fields = true;
    options.preserve_proto_field_names = true;
    auto status = google::protobuf::util::MessageToJsonString(sirProto, &str, options);
    if(!status.ok())
      throw std::runtime_error(format("cannot serialize SIR: %s", status.ToString()));
    break;
  }
  case dawn::SIRSerializer::Format::Byte: {
    if(!sirProto.SerializeToString(&str))
      throw std::runtime_error(dawn::format(
          "cannot deserialize SIR: %s", ProtobufLogger::getInstance().getErrorMessagesAndReset()));
    break;
  }
  }

  return str;
}

void SIRSerializer::serialize(const std::string& file, const SIR* sir, SIRSerializer::Format kind) {
  std::ofstream ofs(file);
  if(!ofs.is_open())
    throw std::runtime_error(format("cannot serialize SIR: failed to open file \"%s\"", file));

  auto str = serializeImpl(sir, kind);
  std::copy(str.begin(), str.end(), std::ostreambuf_iterator<char>(ofs));
}

std::string SIRSerializer::serializeToString(const SIR* sir, SIRSerializer::Format kind) {
  return serializeImpl(sir, kind);
}

//===------------------------------------------------------------------------------------------===//
//     Deserialization
//===------------------------------------------------------------------------------------------===//

namespace {

static std::shared_ptr<sir::AST> makeAST(const dawn::proto::statements::AST& astProto);

template <class T>
static SourceLocation makeLocation(const T& proto) {
  return proto.has_loc() ? SourceLocation(proto.loc().line(), proto.loc().column())
                         : SourceLocation{};
}

static std::shared_ptr<sir::Field> makeField(const dawn::proto::statements::Field& fieldProto) {
  auto field = std::make_shared<sir::Field>(fieldProto.name(),
                                            makeFieldDimensions(fieldProto.field_dimensions()),
                                            makeLocation(fieldProto));
  field->IsTemporary = fieldProto.is_temporary();

  return field;
}

static BuiltinTypeID
makeBuiltinTypeID(const dawn::proto::statements::BuiltinType& builtinTypeProto) {
  switch(builtinTypeProto.type_id()) {
  case dawn::proto::statements::BuiltinType_TypeID_Invalid:
    return BuiltinTypeID::Invalid;
  case dawn::proto::statements::BuiltinType_TypeID_Auto:
    return BuiltinTypeID::Auto;
  case dawn::proto::statements::BuiltinType_TypeID_Boolean:
    return BuiltinTypeID::Boolean;
  case dawn::proto::statements::BuiltinType_TypeID_Integer:
    return BuiltinTypeID::Integer;
  case dawn::proto::statements::BuiltinType_TypeID_Float:
    return BuiltinTypeID::Float;
  case dawn::proto::statements::BuiltinType_TypeID_Double:
    return BuiltinTypeID::Double;
  default:
    return BuiltinTypeID::Invalid;
  }
  return BuiltinTypeID::Invalid;
}

static std::shared_ptr<sir::Direction>
makeDirection(const dawn::proto::statements::Direction& directionProto) {
  return std::make_shared<sir::Direction>(directionProto.name(), makeLocation(directionProto));
}

static std::shared_ptr<sir::Offset> makeOffset(const dawn::proto::statements::Offset& offsetProto) {
  return std::make_shared<sir::Offset>(offsetProto.name(), makeLocation(offsetProto));
}

static std::shared_ptr<sir::Interval>
makeInterval(const dawn::proto::statements::Interval& intervalProto) {
  int lowerLevel = -1, upperLevel = -1, lowerOffset = -1, upperOffset = -1;

  if(intervalProto.LowerLevel_case() == dawn::proto::statements::Interval::kSpecialLowerLevel)
    lowerLevel = intervalProto.special_lower_level() ==
                         dawn::proto::statements::Interval_SpecialLevel::Interval_SpecialLevel_Start
                     ? sir::Interval::Start
                     : sir::Interval::End;
  else
    lowerLevel = intervalProto.lower_level();

  if(intervalProto.UpperLevel_case() == dawn::proto::statements::Interval::kSpecialUpperLevel)
    upperLevel = intervalProto.special_upper_level() ==
                         dawn::proto::statements::Interval_SpecialLevel::Interval_SpecialLevel_Start
                     ? sir::Interval::Start
                     : sir::Interval::End;
  else
    upperLevel = intervalProto.upper_level();

  lowerOffset = intervalProto.lower_offset();
  upperOffset = intervalProto.upper_offset();
  return std::make_shared<sir::Interval>(lowerLevel, upperLevel, lowerOffset, upperOffset);
}

static std::shared_ptr<sir::VerticalRegion>
makeVerticalRegion(const dawn::proto::statements::VerticalRegion& verticalRegionProto) {
  // VerticalRegion.Loc
  auto loc = makeLocation(verticalRegionProto);

  // VerticalRegion.Ast
  auto ast = makeAST(verticalRegionProto.ast());

  // VerticalRegion.VerticalInterval
  auto interval = makeInterval(verticalRegionProto.interval());

  // VerticalRegion.LoopOrder
  auto loopOrder =
      verticalRegionProto.loop_order() == dawn::proto::statements::VerticalRegion::Backward
          ? sir::VerticalRegion::LoopOrderKind::Backward
          : sir::VerticalRegion::LoopOrderKind::Forward;

  auto verticalRegion = std::make_shared<sir::VerticalRegion>(ast, interval, loopOrder, loc);

  if(verticalRegionProto.has_i_range()) {
    verticalRegion->IterationSpace[0] = *makeInterval(verticalRegionProto.i_range());
  }
  if(verticalRegionProto.has_j_range()) {
    verticalRegion->IterationSpace[1] = *makeInterval(verticalRegionProto.j_range());
  }

  return verticalRegion;
}

static std::shared_ptr<ast::StencilCall>
makeStencilCall(const dawn::proto::statements::StencilCall& stencilCallProto) {
  auto stencilCall =
      std::make_shared<ast::StencilCall>(stencilCallProto.callee(), makeLocation(stencilCallProto));

  for(const auto& argName : stencilCallProto.arguments())
    stencilCall->Args.emplace_back(argName);

  return stencilCall;
}

static std::shared_ptr<sir::Expr> makeExpr(const dawn::proto::statements::Expr& expressionProto) {
  switch(expressionProto.expr_case()) {
  case dawn::proto::statements::Expr::kUnaryOperator: {
    const auto& exprProto = expressionProto.unary_operator();
    return std::make_shared<sir::UnaryOperator>(makeExpr(exprProto.operand()), exprProto.op(),
                                                makeLocation(exprProto));
  }
  case dawn::proto::statements::Expr::kBinaryOperator: {
    const auto& exprProto = expressionProto.binary_operator();
    return std::make_shared<sir::BinaryOperator>(makeExpr(exprProto.left()), exprProto.op(),
                                                 makeExpr(exprProto.right()),
                                                 makeLocation(exprProto));
  }
  case dawn::proto::statements::Expr::kAssignmentExpr: {
    const auto& exprProto = expressionProto.assignment_expr();
    return std::make_shared<sir::AssignmentExpr>(makeExpr(exprProto.left()),
                                                 makeExpr(exprProto.right()), exprProto.op(),
                                                 makeLocation(exprProto));
  }
  case dawn::proto::statements::Expr::kTernaryOperator: {
    const auto& exprProto = expressionProto.ternary_operator();
    return std::make_shared<sir::TernaryOperator>(
        makeExpr(exprProto.cond()), makeExpr(exprProto.left()), makeExpr(exprProto.right()),
        makeLocation(exprProto));
  }
  case dawn::proto::statements::Expr::kFunCallExpr: {
    const auto& exprProto = expressionProto.fun_call_expr();
    auto expr = std::make_shared<sir::FunCallExpr>(exprProto.callee(), makeLocation(exprProto));
    for(const auto& argProto : exprProto.arguments())
      expr->getArguments().emplace_back(makeExpr(argProto));
    return expr;
  }
  case dawn::proto::statements::Expr::kStencilFunCallExpr: {
    const auto& exprProto = expressionProto.stencil_fun_call_expr();
    auto expr =
        std::make_shared<sir::StencilFunCallExpr>(exprProto.callee(), makeLocation(exprProto));
    for(const auto& argProto : exprProto.arguments())
      expr->getArguments().emplace_back(makeExpr(argProto));
    return expr;
  }
  case dawn::proto::statements::Expr::kStencilFunArgExpr: {
    const auto& exprProto = expressionProto.stencil_fun_arg_expr();
    int direction = -1, offset = 0, argumentIndex = -1; // default values

    if(exprProto.has_dimension()) {
      switch(exprProto.dimension().direction()) {
      case dawn::proto::statements::Dimension_Direction_I:
        direction = 0;
        break;
      case dawn::proto::statements::Dimension_Direction_J:
        direction = 1;
        break;
      case dawn::proto::statements::Dimension_Direction_K:
        direction = 2;
        break;
      case dawn::proto::statements::Dimension_Direction_Invalid:
      default:
        direction = -1;
        break;
      }
    }
    offset = exprProto.offset();
    argumentIndex = exprProto.argument_index();
    return std::make_shared<sir::StencilFunArgExpr>(direction, offset, argumentIndex,
                                                    makeLocation(exprProto));
  }
  case dawn::proto::statements::Expr::kVarAccessExpr: {
    const auto& exprProto = expressionProto.var_access_expr();
    auto expr = std::make_shared<sir::VarAccessExpr>(
        exprProto.name(), exprProto.has_index() ? makeExpr(exprProto.index()) : nullptr,
        makeLocation(exprProto));
    expr->setIsExternal(exprProto.is_external());
    return expr;
  }
  case dawn::proto::statements::Expr::kFieldAccessExpr: {
    using ProtoFieldAccessExpr = dawn::proto::statements::FieldAccessExpr;

    const auto& exprProto = expressionProto.field_access_expr();
    auto name = exprProto.name();
    auto negateOffset = exprProto.negate_offset();

    auto throwException = [&exprProto](const char* member) {
      throw std::runtime_error(format("FieldAccessExpr::%s (loc %s) exceeds 3 dimensions", member,
                                      makeLocation(exprProto)));
    };

    ast::Offsets offset;
    switch(exprProto.horizontal_offset_case()) {
    case ProtoFieldAccessExpr::kCartesianOffset: {
      auto const& hOffset = exprProto.cartesian_offset();
      offset = ast::Offsets{ast::cartesian, hOffset.i_offset(), hOffset.j_offset(),
                            exprProto.vertical_offset()};
      break;
    }
    case ProtoFieldAccessExpr::kUnstructuredOffset: {
      auto const& hOffset = exprProto.unstructured_offset();
      offset = ast::Offsets{ast::unstructured, hOffset.has_offset(), exprProto.vertical_offset()};
      break;
    }
    case ProtoFieldAccessExpr::kZeroOffset:
      offset = ast::Offsets{ast::HorizontalOffset{}, exprProto.vertical_offset()};
      break;
    default:
      throw std::invalid_argument("Unknown offset");
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

    return std::make_shared<sir::FieldAccessExpr>(name, offset, argumentMap, argumentOffset,
                                                  negateOffset, makeLocation(exprProto));
  }
  case dawn::proto::statements::Expr::kLiteralAccessExpr: {
    const auto& exprProto = expressionProto.literal_access_expr();
    return std::make_shared<sir::LiteralAccessExpr>(
        exprProto.value(), makeBuiltinTypeID(exprProto.type()), makeLocation(exprProto));
  }
  case dawn::proto::statements::Expr::kReductionOverNeighborExpr: {
    const auto& exprProto = expressionProto.reduction_over_neighbor_expr();
    std::vector<std::shared_ptr<sir::Expr>> weights;

    for(const auto& weightProto : exprProto.weights()) {
      weights.push_back(makeExpr(weightProto));
    }

    ast::NeighborChain chain;
    for(int i = 0; i < exprProto.chain_size(); ++i) {
      chain.push_back(getLocationTypeFromProtoLocationType(exprProto.chain(i)));
    }

    if(weights.size() > 0) {
      return std::make_shared<sir::ReductionOverNeighborExpr>(
          exprProto.op(), makeExpr(exprProto.rhs()), makeExpr(exprProto.init()), weights, chain,
          makeLocation(exprProto));
    } else {
      return std::make_shared<sir::ReductionOverNeighborExpr>(
          exprProto.op(), makeExpr(exprProto.rhs()), makeExpr(exprProto.init()), chain,
          makeLocation(exprProto));
    }
  }
  case dawn::proto::statements::Expr::EXPR_NOT_SET:
  default:
    throw std::out_of_range("expr not set");
  }
  return nullptr;
}

static std::shared_ptr<sir::Stmt> makeStmt(const dawn::proto::statements::Stmt& statementProto) {
  switch(statementProto.stmt_case()) {
  case dawn::proto::statements::Stmt::kBlockStmt: {
    const auto& stmtProto = statementProto.block_stmt();
    auto stmt = sir::makeBlockStmt(makeLocation(stmtProto));

    for(const auto& s : stmtProto.statements())
      stmt->push_back(makeStmt(s));

    return stmt;
  }
  case dawn::proto::statements::Stmt::kLoopStmt: {
    const auto& stmtProto = statementProto.loop_stmt();
    const auto& blockStmt = makeStmt(stmtProto.statements());
    DAWN_ASSERT_MSG(blockStmt->getKind() == ast::Stmt::Kind::BlockStmt, "Expected a BlockStmt.");

    switch(stmtProto.loop_descriptor().desc_case()) {
    case dawn::proto::statements::LoopDescriptor::kLoopDescriptorChain: {

      ast::NeighborChain chain;
      for(int i = 0; i < stmtProto.loop_descriptor().loop_descriptor_chain().chain_size(); ++i) {
        chain.push_back(getLocationTypeFromProtoLocationType(
            stmtProto.loop_descriptor().loop_descriptor_chain().chain(i)));
      }
      auto stmt =
          sir::makeLoopStmt(std::move(chain), std::dynamic_pointer_cast<ast::BlockStmt>(blockStmt),
                            makeLocation(stmtProto));
      return stmt;
    }
    case dawn::proto::statements::LoopDescriptor::kLoopDescriptorGeneral: {
      dawn_unreachable("general loop bounds not implemented!\n");
      break;
    }
    default:
      dawn_unreachable("descriptor not set!\n");
    }
  }
  case dawn::proto::statements::Stmt::kExprStmt: {
    const auto& stmtProto = statementProto.expr_stmt();
    return sir::makeExprStmt(makeExpr(stmtProto.expr()), makeLocation(stmtProto));
  }
  case dawn::proto::statements::Stmt::kReturnStmt: {
    const auto& stmtProto = statementProto.return_stmt();
    return sir::makeReturnStmt(makeExpr(stmtProto.expr()), makeLocation(stmtProto));
  }
  case dawn::proto::statements::Stmt::kVarDeclStmt: {
    const auto& stmtProto = statementProto.var_decl_stmt();

    std::vector<std::shared_ptr<sir::Expr>> initList;
    for(const auto& e : stmtProto.init_list())
      initList.emplace_back(makeExpr(e));

    const dawn::proto::statements::Type& typeProto = stmtProto.type();
    CVQualifier cvQual = CVQualifier::Invalid;
    if(typeProto.is_const())
      cvQual |= CVQualifier::Const;
    if(typeProto.is_volatile())
      cvQual |= CVQualifier::Volatile;
    Type type = typeProto.name().empty() ? Type(makeBuiltinTypeID(typeProto.builtin_type()), cvQual)
                                         : Type(typeProto.name(), cvQual);

    return sir::makeVarDeclStmt(type, stmtProto.name(), stmtProto.dimension(),
                                stmtProto.op().c_str(), initList, makeLocation(stmtProto));
  }
  case dawn::proto::statements::Stmt::kStencilCallDeclStmt: {
    const auto& stmtProto = statementProto.stencil_call_decl_stmt();
    return sir::makeStencilCallDeclStmt(makeStencilCall(stmtProto.stencil_call()),
                                        makeLocation(stmtProto));
  }
  case dawn::proto::statements::Stmt::kVerticalRegionDeclStmt: {
    const auto& stmtProto = statementProto.vertical_region_decl_stmt();
    return sir::makeVerticalRegionDeclStmt(makeVerticalRegion(stmtProto.vertical_region()),
                                           makeLocation(stmtProto));
  }
  case dawn::proto::statements::Stmt::kBoundaryConditionDeclStmt: {
    const auto& stmtProto = statementProto.boundary_condition_decl_stmt();
    auto stmt = sir::makeBoundaryConditionDeclStmt(stmtProto.functor(), makeLocation(stmtProto));
    for(const auto& fieldName : stmtProto.fields())
      stmt->getFields().emplace_back(fieldName);
    return stmt;
  }
  case dawn::proto::statements::Stmt::kIfStmt: {
    const auto& stmtProto = statementProto.if_stmt();
    return sir::makeIfStmt(makeStmt(stmtProto.cond_part()), makeStmt(stmtProto.then_part()),
                           stmtProto.has_else_part() ? makeStmt(stmtProto.else_part()) : nullptr,
                           makeLocation(stmtProto));
  }
  case dawn::proto::statements::Stmt::STMT_NOT_SET:
  default:
    throw std::out_of_range("stmt not set");
  }
  return nullptr;
}

static std::shared_ptr<sir::AST> makeAST(const dawn::proto::statements::AST& astProto) {
  auto root = dyn_pointer_cast<sir::BlockStmt>(makeStmt(astProto.root()));
  if(!root)
    throw std::runtime_error("root statement of AST is not a 'BlockStmt'");
  auto ast = std::make_shared<sir::AST>(root);
  return ast;
}

static std::shared_ptr<SIR> deserializeImpl(const std::string& str, SIRSerializer::Format kind) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  using namespace sir;
  ProtobufLogger::init();

  // Decode the string
  sir::proto::SIR sirProto;
  switch(kind) {
  case dawn::SIRSerializer::Format::Json: {
    auto status = google::protobuf::util::JsonStringToMessage(str, &sirProto);
    if(!status.ok())
      throw std::runtime_error(dawn::format("cannot deserialize SIR: %s", status.ToString()));
    break;
  }
  case dawn::SIRSerializer::Format::Byte: {
    if(!sirProto.ParseFromString(str))
      throw std::runtime_error(dawn::format(
          "cannot deserialize SIR: %s", ProtobufLogger::getInstance().getErrorMessagesAndReset()));
    break;
  }
  default:
    throw std::invalid_argument("invalid serialization Kind");
  }

  // Convert protobuf SIR to SIR
  std::shared_ptr<SIR> sir;

  try {

    // SIR.GridType
    switch(sirProto.gridtype()) {
    case dawn::proto::enums::GridType::Cartesian:
      sir = std::make_shared<SIR>(ast::GridType::Cartesian);
      break;
    case dawn::proto::enums::GridType::Unstructured:
      sir = std::make_shared<SIR>(ast::GridType::Unstructured);
      break;
    default:
      throw std::out_of_range("Unknown grid type");
    }

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
      for(const dawn::proto::statements::Field& fieldProto : stencilProto.fields())
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
      for(const dawn::proto::statements::StencilFunctionArg& sirArg :
          stencilFunctionProto.arguments()) {
        switch(sirArg.Arg_case()) {
        case dawn::proto::statements::StencilFunctionArg::kFieldValue:
          stencilFunction->Args.emplace_back(makeField(sirArg.field_value()));
          break;
        case dawn::proto::statements::StencilFunctionArg::kDirectionValue:
          stencilFunction->Args.emplace_back(makeDirection(sirArg.direction_value()));
          break;
        case dawn::proto::statements::StencilFunctionArg::kOffsetValue:
          stencilFunction->Args.emplace_back(makeOffset(sirArg.offset_value()));
          break;
        case dawn::proto::statements::StencilFunctionArg::ARG_NOT_SET:
        default:
          throw std::out_of_range("argument not set");
        }
      }

      // StencilFunction.Intervals
      for(const dawn::proto::statements::Interval& sirInterval : stencilFunctionProto.intervals())
        stencilFunction->Intervals.emplace_back(makeInterval(sirInterval));

      // StencilFunction.Asts
      for(const dawn::proto::statements::AST& sirAst : stencilFunctionProto.asts())
        stencilFunction->Asts.emplace_back(makeAST(sirAst));

      sir->StencilFunctions.emplace_back(stencilFunction);
    }

    // SIR.GlobalVariableMap
    for(const auto& nameValuePair : sirProto.global_variables().map()) {
      const std::string& sirName = nameValuePair.first;
      const sir::proto::GlobalVariableValue& sirValue = nameValuePair.second;
      std::shared_ptr<Global> value = nullptr;
      bool isConstExpr = sirValue.is_constexpr();

      switch(sirValue.Value_case()) {
      case sir::proto::GlobalVariableValue::kBooleanValue:
        value = std::make_shared<Global>(static_cast<bool>(sirValue.boolean_value()), isConstExpr);
        break;
      case sir::proto::GlobalVariableValue::kIntegerValue:
        value = std::make_shared<Global>(static_cast<int>(sirValue.integer_value()), isConstExpr);
        break;
      case sir::proto::GlobalVariableValue::kFloatValue:
        value = std::make_shared<Global>(static_cast<float>(sirValue.float_value()), isConstExpr);
        break;
      case sir::proto::GlobalVariableValue::kDoubleValue:
        value = std::make_shared<Global>(static_cast<double>(sirValue.double_value()), isConstExpr);
        break;
      case sir::proto::GlobalVariableValue::kStringValue:
        value = std::make_shared<Global>(static_cast<std::string>(sirValue.string_value()),
                                         isConstExpr);
        break;
      case sir::proto::GlobalVariableValue::VALUE_NOT_SET:
      default:
        throw std::out_of_range("value not set");
      }

      sir->GlobalVariableMap->emplace(sirName, std::move(*value));
    }

  } catch(std::runtime_error& error) {
    throw std::runtime_error(dawn::format("cannot deserialize SIR: %s", error.what()));
  }
  return sir;
}

} // anonymous namespace

std::shared_ptr<SIR> SIRSerializer::deserialize(const std::string& file,
                                                SIRSerializer::Format kind) {
  std::ifstream ifs(file);
  if(!ifs.is_open())
    throw std::runtime_error(
        dawn::format("cannot deserialize SIR: failed to open file \"%s\"", file));

  std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  return deserializeImpl(str, kind);
}

std::shared_ptr<SIR> SIRSerializer::deserializeFromString(const std::string& str,
                                                          SIRSerializer::Format kind) {
  return deserializeImpl(str, kind);
}

} // namespace dawn
