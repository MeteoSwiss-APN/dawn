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

#include "dawn/AST/ASTStmt.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/AST/ASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include "dawn/AST/AST/statements.pb.h"
#include <stack>

using namespace dawn;

proto::ast::LocationType getProtoLocationTypeFromLocationType(ast::LocationType locationType);

ast::LocationType
getLocationTypeFromProtoLocationType(proto::ast::LocationType protoLocationType);

void setAST(dawn::proto::ast::AST* astProto, const ast::AST* ast);

void setLocation(dawn::proto::ast::SourceLocation* locProto, const SourceLocation& loc);

void setBuiltinType(dawn::proto::ast::BuiltinType* builtinTypeProto,
                    const BuiltinTypeID& builtinType);

void setInterval(dawn::proto::ast::Interval* intervalProto, const ast::Interval* interval);

void setDirection(dawn::proto::ast::Direction* directionProto,
                  const sir::Direction* direction);

void setOffset(dawn::proto::ast::Offset* offsetProto, const sir::Offset* offset);

void setFieldDimensions(dawn::proto::ast::FieldDimensions* protoFieldDimensions,
                        const ast::FieldDimensions& fieldDimensions);

void setField(dawn::proto::ast::Field* fieldProto, const sir::Field* field);

dawn::proto::ast::Extents makeProtoExtents(dawn::iir::Extents const& extents);

void setAccesses(dawn::proto::ast::Accesses* protoAccesses,
                 const std::optional<iir::Accesses>& accesses);

iir::Extents makeExtents(const dawn::proto::ast::Extents* protoExtents);

class ProtoStmtBuilder : public ast::ASTVisitorNonConst {
  std::stack<dawn::proto::ast::Stmt*> currentStmtProto_;
  std::stack<dawn::proto::ast::Expr*> currentExprProto_;
  const dawn::ast::StmtData::DataType dataType_;

public:
  ProtoStmtBuilder(dawn::proto::ast::Stmt* stmtProto,
                   dawn::ast::StmtData::DataType dataType);

  ProtoStmtBuilder(dawn::proto::ast::Expr* exprProto,
                   dawn::ast::StmtData::DataType dataType);

  dawn::proto::ast::Stmt* getCurrentStmtProto();

  dawn::proto::ast::Expr* getCurrentExprProto();

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

void setAST(proto::ast::AST* astProto, const ast::AST* ast);

//===------------------------------------------------------------------------------------------===//
// Deserialization
//===------------------------------------------------------------------------------------------===//

template <class T>
SourceLocation makeLocation(const T& proto) {
  return proto.has_loc() ? SourceLocation(proto.loc().line(), proto.loc().column())
                         : SourceLocation{};
}

ast::FieldDimensions
makeFieldDimensions(const proto::ast::FieldDimensions& protoFieldDimensions);

std::shared_ptr<sir::Field> makeField(const proto::ast::Field& fieldProto);

BuiltinTypeID makeBuiltinTypeID(const proto::ast::BuiltinType& builtinTypeProto);

std::shared_ptr<sir::Direction> makeDirection(const proto::ast::Direction& directionProto);

std::shared_ptr<sir::Offset> makeOffset(const proto::ast::Offset& offsetProto);

std::shared_ptr<ast::Interval> makeInterval(const proto::ast::Interval& intervalProto);

std::shared_ptr<ast::Expr> makeExpr(const proto::ast::Expr& expressionProto,
                                    ast::StmtData::DataType dataType, int& maxID);

std::shared_ptr<ast::Stmt> makeStmt(const proto::ast::Stmt& statementProto,
                                    ast::StmtData::DataType dataType, int& maxID);

std::shared_ptr<ast::AST> makeAST(const dawn::proto::ast::AST& astProto,
                                  ast::StmtData::DataType dataType, int& maxID);
