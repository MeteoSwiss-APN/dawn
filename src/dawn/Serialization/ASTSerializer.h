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

#ifndef DAWN_SUPPORT_ASTSERIALIZER_H
#define DAWN_SUPPORT_ASTSERIALIZER_H

#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include "dawn/SIR/SIR/statements.pb.h"
#include <stack>

using namespace dawn;

void setAST(dawn::proto::statements::AST* astProto, const AST* ast);

void setLocation(dawn::proto::statements::SourceLocation* locProto, const SourceLocation& loc);

void setBuiltinType(dawn::proto::statements::BuiltinType* builtinTypeProto,
                    const BuiltinTypeID& builtinType);

void setInterval(dawn::proto::statements::Interval* intervalProto, const sir::Interval* interval);

void setDirection(dawn::proto::statements::Direction* directionProto,
                  const sir::Direction* direction);

void setOffset(dawn::proto::statements::Offset* offsetProto, const sir::Offset* offset);

void setField(dawn::proto::statements::Field* fieldProto, const sir::Field* field);

class ProtoStmtBuilder : public dawn::ASTVisitor {
  std::stack<dawn::proto::statements::Stmt*> currentStmtProto_;
  std::stack<dawn::proto::statements::Expr*> currentExprProto_;

public:
  ProtoStmtBuilder(dawn::proto::statements::Stmt* stmtProto);

  ProtoStmtBuilder(dawn::proto::statements::Expr* exprProto);

  dawn::proto::statements::Stmt* getCurrentStmtProto();

  dawn::proto::statements::Expr* getCurrentExprProto();

  void visit(const std::shared_ptr<BlockStmt>& stmt) override;

  void visit(const std::shared_ptr<ExprStmt>& stmt) override;

  void visit(const std::shared_ptr<ReturnStmt>& stmt) override;
  void visit(const std::shared_ptr<VarDeclStmt>& stmt) override;

  void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) override;

  void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) override;

  void visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) override;

  void visit(const std::shared_ptr<IfStmt>& stmt) override;

  void visit(const std::shared_ptr<UnaryOperator>& expr) override;

  void visit(const std::shared_ptr<BinaryOperator>& expr) override;

  void visit(const std::shared_ptr<AssignmentExpr>& expr) override;

  void visit(const std::shared_ptr<TernaryOperator>& expr) override;

  void visit(const std::shared_ptr<FunCallExpr>& expr) override;

  void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override;

  void visit(const std::shared_ptr<StencilFunArgExpr>& expr) override;

  void visit(const std::shared_ptr<VarAccessExpr>& expr) override;

  void visit(const std::shared_ptr<FieldAccessExpr>& expr) override;

  void visit(const std::shared_ptr<LiteralAccessExpr>& expr) override;
};

void setAST(proto::statements::AST* astProto, const AST* ast);

//===------------------------------------------------------------------------------------------===//
// Deserialization
//===------------------------------------------------------------------------------------------===//

template <class T>
SourceLocation makeLocation(const T& proto) {
  return proto.has_loc() ? SourceLocation(proto.loc().line(), proto.loc().column())
                         : SourceLocation{};
}

std::shared_ptr<sir::Field> makeField(const proto::statements::Field& fieldProto);

BuiltinTypeID makeBuiltinTypeID(const proto::statements::BuiltinType& builtinTypeProto);

std::shared_ptr<sir::Direction> makeDirection(const proto::statements::Direction& directionProto);

std::shared_ptr<sir::Offset> makeOffset(const proto::statements::Offset& offsetProto);

std::shared_ptr<sir::Interval> makeInterval(const proto::statements::Interval& intervalProto);

std::shared_ptr<Expr> makeExpr(const proto::statements::Expr& expressionProto);

std::shared_ptr<Stmt> makeStmt(const proto::statements::Stmt& statementProto);

std::shared_ptr<AST> makeAST(const dawn::proto::statements::AST& astProto);

#endif // DAWN_SUPPORT_ASTSERIALIZER_H
