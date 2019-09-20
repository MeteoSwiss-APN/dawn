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

enum class field_type { ijk, ij };
static Array3i as_array(field_type ft) {
  switch(ft) {
  case field_type::ijk:
    return Array3i{0, 0, 0};
  case field_type::ij:
    return Array3i{0, 0, 1};
  }
  return {};
}

enum class op { multiply, plus, minus, reduce_over_neighbor, assign };
enum class access_type { r, rw };
class IIRBuilder {
public:
  std::shared_ptr<iir::Expr> make_reduce_over_neighbor_expr(op operation,
                                                            std::shared_ptr<iir::Expr> const& rhs,
                                                            std::shared_ptr<iir::Expr> const& init);

  std::shared_ptr<iir::Expr> make_multiply_expr(std::shared_ptr<iir::Expr> const& lhs,
                                                std::shared_ptr<iir::Expr> const& rhs);

  std::shared_ptr<iir::Expr> make_assign_expr(std::shared_ptr<iir::Expr> const& lhs,
                                              std::shared_ptr<iir::Expr> const& rhs,
                                              op operation = op::assign);

  int make_field(std::string const& name, field_type ft = field_type::ijk);

  template <typename T>
  std::shared_ptr<iir::Expr> make_lit(T&& v) {
    int acc =
        si_->getMetaData().insertAccessOfType(iir::FieldAccessType::FAT_Literal, std::to_string(v));
    auto expr = std::make_shared<iir::LiteralAccessExpr>(
        std::to_string(v),
        sir::Value::typeToBuiltinTypeID(sir::Value::TypeInfo<typename std::decay<T>::type>::Type));
    expr->setID(-si_->nextUID());
    si_->getMetaData().insertExprToAccessID(expr, acc);
    return expr;
  }

  std::shared_ptr<iir::Expr> at(int field_id, access_type access = access_type::r,
                                Array3i extent = {});

  std::shared_ptr<iir::Expr> at(int field_id, Array3i extent);

  std::unique_ptr<iir::StatementAccessesPair> make_stmt(std::shared_ptr<iir::Expr>&& expr);

  template <typename... Stmts>
  std::unique_ptr<iir::DoMethod> make_do(sir::Interval::LevelKind s, sir::Interval::LevelKind e,
                                         Stmts&&... stmts) {
    auto ret = make_unique<iir::DoMethod>(iir::Interval(s, e), si_->getMetaData());
    ret->setID(si_->nextUID());
    int x[] = {(ret->insertChild(std::forward<Stmts>(stmts)), ret->updateLevel(), 0)...};
    (void*)x;
    return ret;
  }
  template <typename... DoMethods>
  std::unique_ptr<iir::Stage> make_stage(DoMethods&&... do_methods) {
    auto ret = make_unique<iir::Stage>(si_->getMetaData(), si_->nextUID());
    int x[] = {(ret->insertChild(std::forward<DoMethods>(do_methods)), 0)...};
    (void*)x;
    return ret;
  }
  template <typename... Stages>
  std::unique_ptr<iir::MultiStage> make_multistage(iir::LoopOrderKind loop_kind,
                                                   Stages&&... stages) {
    auto ret = make_unique<iir::MultiStage>(si_->getMetaData(), loop_kind);
    ret->setID(si_->nextUID());
    int x[] = {(ret->insertChild(std::forward<Stages>(stages)), 0)...};
    (void*)x;
    return ret;
  }
  template <typename... MultiStages>
  std::unique_ptr<iir::Stencil> make_stencil(MultiStages&&... multistages) {
    auto ret = make_unique<iir::Stencil>(si_->getMetaData(), sir::Attr{}, si_->nextUID());
    int x[] = {(ret->insertChild(std::forward<MultiStages>(multistages)), 0)...};
    (void*)x;
    return ret;
  }

  std::shared_ptr<iir::StencilInstantiation> build(std::string const& name,
                                                   std::unique_ptr<iir::Stencil> stencil);

  IIRBuilder() : si_(std::make_shared<iir::StencilInstantiation>()) {}

private:
  std::shared_ptr<iir::StencilInstantiation> si_;
  std::map<int, std::string> field_names_;
  std::map<std::string, int> field_ids_;
  std::map<iir::Expr*, Array3i> read_extents_;
  std::map<iir::Expr*, Array3i> write_extents_;
};
} // namespace iir
} // namespace dawn

#endif
