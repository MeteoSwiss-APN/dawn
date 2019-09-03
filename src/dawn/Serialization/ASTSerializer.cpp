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
#include "dawn/SIR/ASTStmt.h"
#include <fstream>
#include <google/protobuf/util/json_util.h>
#include <list>
#include <tuple>
#include <utility>

namespace dawn {
namespace iir {
struct IIRASTData;
}
namespace sir {
struct SIRASTData;
}
} // namespace dawn

using namespace dawn;
using namespace ast;

template <>
void setAST<sir::SIRASTData>(proto::statements::AST* astProto, const sir::AST* ast);

template <>
void setAST<iir::IIRASTData>(proto::statements::AST* astProto, const iir::AST* ast);

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

void SIRProtoStmtBuilder::visit(const std::shared_ptr<sir::VerticalRegionDeclStmt>& stmt) {
  auto protoStmt = getCurrentStmtProto()->mutable_vertical_region_decl_stmt();

  // VerticalRegionDeclStmt.Ast
  setAST(protoStmt->mutable_ast(), stmt->getAST().get());

  dawn::sir::VerticalRegion* verticalRegion = stmt->verticalRegion_.get();
  dawn::proto::statements::VerticalRegion* verticalRegionProto =
      protoStmt->mutable_vertical_region();

  // VerticalRegion.Loc
  setLocation(verticalRegionProto->mutable_loc(), verticalRegion->Loc);

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

template <>
void setAST<sir::SIRASTData>(proto::statements::AST* astProto, const sir::AST* ast) {
  SIRProtoStmtBuilder builder(astProto->mutable_root());
  ast->accept(builder);
}

template <>
void setAST<iir::IIRASTData>(proto::statements::AST* astProto, const iir::AST* ast) {
  IIRProtoStmtBuilder builder(astProto->mutable_root());
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

template <typename DataTraits>
std::shared_ptr<Stmt<DataTraits>> makeStmtImpl(const proto::statements::Stmt& statementProto) {
  switch(statementProto.stmt_case()) {
  case proto::statements::Stmt::kBlockStmt: {
    const auto& stmtProto = statementProto.block_stmt();
    auto stmt = std::make_shared<BlockStmt<DataTraits>>(makeLocation(stmtProto));

    for(const auto& s : stmtProto.statements())
      stmt->push_back(makeStmt<DataTraits>(s));
    stmt->setID(stmtProto.id());

    return stmt;
  }
  case proto::statements::Stmt::kExprStmt: {
    const auto& stmtProto = statementProto.expr_stmt();
    auto stmt = std::make_shared<ExprStmt<DataTraits>>(makeExpr<DataTraits>(stmtProto.expr()),
                                                       makeLocation(stmtProto));
    stmt->setID(stmtProto.id());
    return stmt;
  }
  case proto::statements::Stmt::kReturnStmt: {
    const auto& stmtProto = statementProto.return_stmt();
    auto stmt = std::make_shared<ReturnStmt<DataTraits>>(makeExpr<DataTraits>(stmtProto.expr()),
                                                         makeLocation(stmtProto));
    stmt->setID(stmtProto.id());
    return stmt;
  }
  case proto::statements::Stmt::kVarDeclStmt: {
    const auto& stmtProto = statementProto.var_decl_stmt();

    std::vector<std::shared_ptr<Expr<DataTraits>>> initList;
    for(const auto& e : stmtProto.init_list())
      initList.emplace_back(makeExpr<DataTraits>(e));

    const proto::statements::Type& typeProto = stmtProto.type();
    CVQualifier cvQual = CVQualifier::Invalid;
    if(typeProto.is_const())
      cvQual |= CVQualifier::Const;
    if(typeProto.is_volatile())
      cvQual |= CVQualifier::Volatile;
    Type type = typeProto.name().empty() ? Type(makeBuiltinTypeID(typeProto.builtin_type()), cvQual)
                                         : Type(typeProto.name(), cvQual);

    auto stmt = std::make_shared<VarDeclStmt<DataTraits>>(
        type, stmtProto.name(), stmtProto.dimension(), stmtProto.op().c_str(), initList,
        makeLocation(stmtProto));
    stmt->setID(stmtProto.id());
    return stmt;
  }
  case proto::statements::Stmt::kStencilCallDeclStmt: {
    auto metaloc = makeLocation(statementProto.stencil_call_decl_stmt());
    const auto& stmtProto = statementProto.stencil_call_decl_stmt();
    auto loc = makeLocation(stmtProto.stencil_call());
    std::shared_ptr<ast::StencilCall> call =
        std::make_shared<ast::StencilCall>(stmtProto.stencil_call().callee(), loc);
    for(const auto& argName : stmtProto.stencil_call().arguments()) {
      call->Args.push_back(argName);
    }
    auto stmt = std::make_shared<StencilCallDeclStmt<DataTraits>>(call, metaloc);
    stmt->setID(stmtProto.id());
    return stmt;
  }
  case proto::statements::Stmt::kBoundaryConditionDeclStmt: {
    const auto& stmtProto = statementProto.boundary_condition_decl_stmt();
    auto stmt = std::make_shared<BoundaryConditionDeclStmt<DataTraits>>(stmtProto.functor(),
                                                                        makeLocation(stmtProto));
    for(const auto& fieldName : stmtProto.fields())
      stmt->getFields().emplace_back(fieldName);
    stmt->setID(stmtProto.id());
    return stmt;
  }
  case proto::statements::Stmt::kIfStmt: {
    const auto& stmtProto = statementProto.if_stmt();
    auto stmt = std::make_shared<IfStmt<DataTraits>>(
        makeStmt<DataTraits>(stmtProto.cond_part()), makeStmt<DataTraits>(stmtProto.then_part()),
        stmtProto.has_else_part() ? makeStmt<DataTraits>(stmtProto.else_part()) : nullptr,
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

template <>
std::shared_ptr<Stmt<iir::IIRASTData>> makeStmt(const proto::statements::Stmt& statementProto) {
  return makeStmtImpl<iir::IIRASTData>(statementProto);
}

template <>
std::shared_ptr<Stmt<sir::SIRASTData>> makeStmt(const proto::statements::Stmt& statementProto) {
  if(statementProto.stmt_case() == proto::statements::Stmt::kVerticalRegionDeclStmt) {
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
    auto ast = makeAST<sir::SIRASTData>(stmtProto.ast());
    std::shared_ptr<sir::VerticalRegion> verticalRegion =
        std::make_shared<sir::VerticalRegion>(interval, looporder, loc);
    auto stmt = std::make_shared<sir::VerticalRegionDeclStmt>(ast, verticalRegion, loc);
    stmt->setID(stmtProto.id());
    return stmt;
  } else
    return makeStmtImpl<sir::SIRASTData>(statementProto);
}
