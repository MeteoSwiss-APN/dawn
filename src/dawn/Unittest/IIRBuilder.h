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
//
#ifndef DAWN_IIR_IIRBUILDER_H
#define DAWN_IIR_IIRBUILDER_H

#include "dawn/CodeGen/CodeGen.h"
#include "dawn/IIR/ASTStmt.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/Stage.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/AccessComputation.h"

namespace dawn {
namespace iir {

enum class field_type { ijk, ij, ik, jk, i, j, k };

enum class op {
  multiply,
  plus,
  minus,
  assign,
  divide,
  equal,
  not_equal,
  greater,
  less,
  greater_equal,
  less_equal,
  logical_and,
  logical_or,
  logical_not
};
enum class access_type { r, rw };
class IIRBuilder {
  struct Field {
    int id;
    std::string name;
  };
  struct LocalVar {
    int id;
    std::string name;
    std::shared_ptr<VarDeclStmt> decl;
  };
  struct StmtData {
    std::shared_ptr<Stmt> stmt;
    std::unique_ptr<StatementAccessesPair> sap;
  };

public:
  std::shared_ptr<iir::Expr> binaryExpr(std::shared_ptr<iir::Expr> const& lhs,
                                        std::shared_ptr<iir::Expr> const& rhs, op operation);

  std::shared_ptr<iir::Expr> assignExpr(std::shared_ptr<iir::Expr> const& lhs,
                                        std::shared_ptr<iir::Expr> const& rhs,
                                        op operation = op::assign);

  std::shared_ptr<iir::Expr> unaryExpr(std::shared_ptr<iir::Expr> const& expr, op operation);

  std::shared_ptr<iir::Expr> conditionalExpr(std::shared_ptr<iir::Expr> const& cond,
                                             std::shared_ptr<iir::Expr> const& case_then,
                                             std::shared_ptr<iir::Expr> const& case_else);

  Field field(std::string const& name, field_type ft = field_type::ijk);
  LocalVar localvar(std::string const& name);

  template <typename T>
  std::shared_ptr<iir::Expr> lit(T&& v) {
    int acc =
        si_->getMetaData().insertAccessOfType(iir::FieldAccessType::FAT_Literal, std::to_string(v));
    auto expr = std::make_shared<iir::LiteralAccessExpr>(
        std::to_string(v),
        sir::Value::typeToBuiltinTypeID(sir::Value::TypeInfo<typename std::decay<T>::type>::Type));
    expr->setID(-si_->nextUID());
    si_->getMetaData().insertExprToAccessID(expr, acc);
    return expr;
  }

  std::shared_ptr<iir::Expr> at(Field field, access_type access = access_type::r,
                                Array3i extent = {});

  std::shared_ptr<iir::Expr> at(Field field, Array3i extent);

  std::shared_ptr<iir::Expr> at(LocalVar var);

  StmtData stmt(std::shared_ptr<iir::Expr>&& expr);
  template <typename... Stmts>
  StmtData block(Stmts&&... stmts) {
    auto stmt =
        std::make_shared<iir::BlockStmt>(std::vector<std::shared_ptr<iir::Stmt>>{stmts.stmt...});
    auto sap = make_unique<iir::StatementAccessesPair>(std::make_shared<Statement>(stmt, nullptr));
    int x[] = {(stmts.sap ? (sap->insertBlockStatement(std::move(stmts.sap)), 0) : 0)...};
    (void)x;
    return {std::move(stmt), std::move(sap)};
  }
  StmtData ifStmt(std::shared_ptr<iir::Expr>&& cond, StmtData&& case_then,
                  StmtData&& case_else = {nullptr, {}});
  StmtData declareVar(LocalVar& var_id);

  template <typename... Stmts>
  std::unique_ptr<iir::DoMethod> vregion(sir::Interval::LevelKind s, sir::Interval::LevelKind e,
                                         Stmts&&... stmts) {
    auto ret = make_unique<iir::DoMethod>(iir::Interval(s, e), si_->getMetaData());
    ret->setID(si_->nextUID());
    int x[] = {(DAWN_ASSERT(stmts.sap), ret->insertChild(std::move(stmts.sap)), 0)...};
    (void)x;
    computeAccesses(si_.get(), ret->getChildren());
    ret->updateLevel();
    return ret;
  }
  template <typename... DoMethods>
  std::unique_ptr<iir::Stage> stage(DoMethods&&... do_methods) {
    auto ret = make_unique<iir::Stage>(si_->getMetaData(), si_->nextUID());
    int x[] = {(ret->insertChild(std::forward<DoMethods>(do_methods)), 0)...};
    (void)x;
    return ret;
  }
  template <typename... Stages>
  std::unique_ptr<iir::MultiStage> multistage(iir::LoopOrderKind loop_kind, Stages&&... stages) {
    auto ret = make_unique<iir::MultiStage>(si_->getMetaData(), loop_kind);
    ret->setID(si_->nextUID());
    int x[] = {(ret->insertChild(std::forward<Stages>(stages)), 0)...};
    (void)x;
    return ret;
  }
  template <typename... MultiStages>
  std::unique_ptr<iir::Stencil> stencil(MultiStages&&... multistages) {
    auto ret = make_unique<iir::Stencil>(si_->getMetaData(), sir::Attr{}, si_->nextUID());
    int x[] = {(ret->insertChild(std::forward<MultiStages>(multistages)), 0)...};
    (void)x;
    return ret;
  }

  dawn::codegen::stencilInstantiationContext build(std::string const& name,
                                                   std::unique_ptr<iir::Stencil> stencil);

  IIRBuilder() : si_(std::make_shared<iir::StencilInstantiation>()) {}

private:
  std::shared_ptr<iir::StencilInstantiation> si_;
};
} // namespace iir
} // namespace dawn

#endif
