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

#include "ASTStmt.h"
#include "MultiStage.h"
#include "Stage.h"
#include "Stencil.h"
#include "StencilInstantiation.h"

#include "dawn/Optimizer/AccessComputation.h"

namespace dawn {
namespace iir {

enum class field_type { ijk, ij, ik, jk, i, j, k };

enum class op {
  reduce_over_neighbor,
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

public:
  std::shared_ptr<iir::Expr> reduce_over_neighbor_expr(op operation,
                                                       std::shared_ptr<iir::Expr> const& rhs,
                                                       std::shared_ptr<iir::Expr> const& init);

  std::shared_ptr<iir::Expr> binary_expr(std::shared_ptr<iir::Expr> const& lhs,
                                         std::shared_ptr<iir::Expr> const& rhs, op operation);

  std::shared_ptr<iir::Expr> assign_expr(std::shared_ptr<iir::Expr> const& lhs,
                                         std::shared_ptr<iir::Expr> const& rhs,
                                         op operation = op::assign);

  std::shared_ptr<iir::Expr> unary_expr(std::shared_ptr<iir::Expr> const& expr, op operation);

  std::shared_ptr<iir::Expr> conditional_expr(std::shared_ptr<iir::Expr> const& cond,
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

  std::shared_ptr<iir::Stmt> stmt(std::shared_ptr<iir::Expr>&& expr);
  std::shared_ptr<iir::Stmt> if_stmt(std::shared_ptr<iir::Expr>&& cond,
                                     std::shared_ptr<iir::Stmt>&& case_then,
                                     std::shared_ptr<iir::Stmt>&& case_else = nullptr);
  std::shared_ptr<iir::Stmt> declare_var(LocalVar& var_id);

  template <typename... Stmts>
  std::unique_ptr<iir::DoMethod> vregion(sir::Interval::LevelKind s, sir::Interval::LevelKind e,
                                         Stmts&&... stmts) {
    auto ret = make_unique<iir::DoMethod>(iir::Interval(s, e), si_->getMetaData());
    ret->setID(si_->nextUID());
    int x[] = {(ret->insertChild(make_unique<iir::StatementAccessesPair>(
                    std::make_shared<Statement>(std::forward<Stmts>(stmts), nullptr))),
                0)...};
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

  std::shared_ptr<iir::StencilInstantiation> build(std::string const& name,
                                                   std::unique_ptr<iir::Stencil> stencil);

  IIRBuilder() : si_(std::make_shared<iir::StencilInstantiation>()) {}

private:
  std::shared_ptr<iir::StencilInstantiation> si_;
};
} // namespace iir
} // namespace dawn

#endif
