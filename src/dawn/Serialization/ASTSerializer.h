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

#include "dawn/AST/ASTFwd.h"
#include "dawn/AST/ASTVisitor.h"
#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include "dawn/SIR/SIR/statements.pb.h"
#include <stack>

using namespace dawn;

template <typename DataTraits>
void setAST(dawn::proto::statements::AST* astProto, const ast::AST<DataTraits>* ast);

void setLocation(dawn::proto::statements::SourceLocation* locProto, const SourceLocation& loc);

void setBuiltinType(dawn::proto::statements::BuiltinType* builtinTypeProto,
                    const BuiltinTypeID& builtinType);

void setInterval(dawn::proto::statements::Interval* intervalProto, const sir::Interval* interval);

void setDirection(dawn::proto::statements::Direction* directionProto,
                  const sir::Direction* direction);

void setOffset(dawn::proto::statements::Offset* offsetProto, const sir::Offset* offset);

void setField(dawn::proto::statements::Field* fieldProto, const sir::Field* field);

template <typename DataTraits>
class ProtoStmtBuilder : virtual public ast::ASTVisitor<DataTraits> {
protected:
  std::stack<dawn::proto::statements::Stmt*> currentStmtProto_;
  std::stack<dawn::proto::statements::Expr*> currentExprProto_;

public:
  ProtoStmtBuilder(dawn::proto::statements::Stmt* stmtProto);

  ProtoStmtBuilder(dawn::proto::statements::Expr* exprProto);

  dawn::proto::statements::Stmt* getCurrentStmtProto();

  dawn::proto::statements::Expr* getCurrentExprProto();

  void visit(const std::shared_ptr<ast::BlockStmt<DataTraits>>& stmt) override;
  void visit(const std::shared_ptr<ast::ExprStmt<DataTraits>>& stmt) override;
  void visit(const std::shared_ptr<ast::ReturnStmt<DataTraits>>& stmt) override;
  void visit(const std::shared_ptr<ast::VarDeclStmt<DataTraits>>& stmt) override;
  void visit(const std::shared_ptr<ast::StencilCallDeclStmt<DataTraits>>& stmt) override;
  void visit(const std::shared_ptr<ast::BoundaryConditionDeclStmt<DataTraits>>& stmt) override;
  void visit(const std::shared_ptr<ast::IfStmt<DataTraits>>& stmt) override;
  void visit(const std::shared_ptr<ast::UnaryOperator<DataTraits>>& expr) override;
  void visit(const std::shared_ptr<ast::BinaryOperator<DataTraits>>& expr) override;
  void visit(const std::shared_ptr<ast::AssignmentExpr<DataTraits>>& expr) override;
  void visit(const std::shared_ptr<ast::TernaryOperator<DataTraits>>& expr) override;
  void visit(const std::shared_ptr<ast::FunCallExpr<DataTraits>>& expr) override;
  void visit(const std::shared_ptr<ast::StencilFunCallExpr<DataTraits>>& expr) override;
  void visit(const std::shared_ptr<ast::StencilFunArgExpr<DataTraits>>& expr) override;
  void visit(const std::shared_ptr<ast::VarAccessExpr<DataTraits>>& expr) override;
  void visit(const std::shared_ptr<ast::FieldAccessExpr<DataTraits>>& expr) override;
  void visit(const std::shared_ptr<ast::LiteralAccessExpr<DataTraits>>& expr) override;
};

using IIRProtoStmtBuilder = ProtoStmtBuilder<iir::IIRASTData>;

class SIRProtoStmtBuilder : ProtoStmtBuilder<sir::SIRASTData>, public sir::ASTVisitor {
public:
  SIRProtoStmtBuilder(dawn::proto::statements::Stmt* stmtProto)
      : ProtoStmtBuilder<sir::SIRASTData>(stmtProto) {}
  SIRProtoStmtBuilder(dawn::proto::statements::Expr* exprProto)
      : ProtoStmtBuilder<sir::SIRASTData>(exprProto) {}

  void visit(const std::shared_ptr<sir::VerticalRegionDeclStmt>& stmt) override;
};

template <typename DataTraits>
void setAST(proto::statements::AST* astProto, const ast::AST<DataTraits>* ast);

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

template <typename DataTraits>
std::shared_ptr<ast::Expr<DataTraits>> makeExpr(const proto::statements::Expr& expressionProto);

template <typename DataTraits>
std::shared_ptr<ast::Stmt<DataTraits>> makeStmt(const proto::statements::Stmt& statementProto);

template <typename DataTraits>
std::shared_ptr<ast::AST<DataTraits>> makeAST(const dawn::proto::statements::AST& astProto);

#include "dawn/Serialization/ASTSerializer.tcc"

#endif // DAWN_SUPPORT_ASTSERIALIZER_H
