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

#pragma once

#include "dawn/AST/ASTFwd.h"
#include "dawn/AST/ASTStmt.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include "dawn/SIR/SIR/statements.pb.h"
#include <stack>

using namespace dawn;

proto::enums::LocationType getProtoLocationTypeFromLocationType(ast::LocationType locationType);

ast::LocationType
getLocationTypeFromProtoLocationType(proto::enums::LocationType protoLocationType);

void setAST(dawn::proto::statements::AST* astProto, const ast::AST* ast);

void setLocation(dawn::proto::statements::SourceLocation* locProto, const SourceLocation& loc);

void setBuiltinType(dawn::proto::statements::BuiltinType* builtinTypeProto,
                    const BuiltinTypeID& builtinType);

void setInterval(dawn::proto::statements::Interval* intervalProto, const sir::Interval* interval);

void setDirection(dawn::proto::statements::Direction* directionProto,
                  const sir::Direction* direction);

void setOffset(dawn::proto::statements::Offset* offsetProto, const sir::Offset* offset);

void setFieldDimensions(dawn::proto::statements::FieldDimensions* protoFieldDimensions,
                        const sir::FieldDimensions& fieldDimensions);

void setField(dawn::proto::statements::Field* fieldProto, const sir::Field* field);

dawn::proto::statements::Extents makeProtoExtents(dawn::iir::Extents const& extents);

void setAccesses(dawn::proto::statements::Accesses* protoAccesses,
                 const std::optional<iir::Accesses>& accesses);

iir::Extents makeExtents(const dawn::proto::statements::Extents* protoExtents);

class ProtoStmtBuilder : public ast::ASTVisitor {
  std::stack<dawn::proto::statements::Stmt*> currentStmtProto_;
  std::stack<dawn::proto::statements::Expr*> currentExprProto_;
  const dawn::ast::StmtData::DataType dataType_;

public:
  ProtoStmtBuilder(dawn::proto::statements::Stmt* stmtProto,
                   dawn::ast::StmtData::DataType dataType);

  ProtoStmtBuilder(dawn::proto::statements::Expr* exprProto,
                   dawn::ast::StmtData::DataType dataType);

  dawn::proto::statements::Stmt* getCurrentStmtProto();

  dawn::proto::statements::Expr* getCurrentExprProto();

  void visit(const std::shared_ptr<ast::BlockStmt>& stmt) override;

  void visit(const std::shared_ptr<ast::LoopStmt>& stmt) override;

  void visit(const std::shared_ptr<ast::ExprStmt>& stmt) override;

  void visit(const std::shared_ptr<ast::ReturnStmt>& stmt) override;
  void visit(const std::shared_ptr<ast::VarDeclStmt>& stmt) override;

  void visit(const std::shared_ptr<ast::VerticalRegionDeclStmt>& stmt) override;

  void visit(const std::shared_ptr<ast::StencilCallDeclStmt>& stmt) override;

  void visit(const std::shared_ptr<ast::BoundaryConditionDeclStmt>& stmt) override;

  void visit(const std::shared_ptr<ast::IfStmt>& stmt) override;

  void visit(const std::shared_ptr<ast::UnaryOperator>& expr) override;

  void visit(const std::shared_ptr<ast::BinaryOperator>& expr) override;

  void visit(const std::shared_ptr<ast::AssignmentExpr>& expr) override;

  void visit(const std::shared_ptr<ast::TernaryOperator>& expr) override;

  void visit(const std::shared_ptr<ast::FunCallExpr>& expr) override;

  void visit(const std::shared_ptr<ast::StencilFunCallExpr>& expr) override;

  void visit(const std::shared_ptr<ast::StencilFunArgExpr>& expr) override;

  void visit(const std::shared_ptr<ast::VarAccessExpr>& expr) override;

  void visit(const std::shared_ptr<ast::FieldAccessExpr>& expr) override;

  void visit(const std::shared_ptr<ast::LiteralAccessExpr>& expr) override;

  void visit(const std::shared_ptr<ast::ReductionOverNeighborExpr>& expr) override;
};

void setAST(proto::statements::AST* astProto, const ast::AST* ast);

//===------------------------------------------------------------------------------------------===//
// Deserialization
//===------------------------------------------------------------------------------------------===//

template <class T>
SourceLocation makeLocation(const T& proto) {
  return proto.has_loc() ? SourceLocation(proto.loc().line(), proto.loc().column())
                         : SourceLocation{};
}

sir::FieldDimensions
makeFieldDimensions(const proto::statements::FieldDimensions& protoFieldDimensions);

std::shared_ptr<sir::Field> makeField(const proto::statements::Field& fieldProto);

BuiltinTypeID makeBuiltinTypeID(const proto::statements::BuiltinType& builtinTypeProto);

std::shared_ptr<sir::Direction> makeDirection(const proto::statements::Direction& directionProto);

std::shared_ptr<sir::Offset> makeOffset(const proto::statements::Offset& offsetProto);

std::shared_ptr<sir::Interval> makeInterval(const proto::statements::Interval& intervalProto);

std::shared_ptr<ast::Expr> makeExpr(const proto::statements::Expr& expressionProto,
                                    ast::StmtData::DataType dataType, int& maxID);

std::shared_ptr<ast::Stmt> makeStmt(const proto::statements::Stmt& statementProto,
                                    ast::StmtData::DataType dataType, int& maxID);

std::shared_ptr<ast::AST> makeAST(const dawn::proto::statements::AST& astProto,
                                  ast::StmtData::DataType dataType, int& maxID);
