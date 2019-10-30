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
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTStmt.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/Stage.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/AccessComputation.h"

namespace dawn {
namespace iir {

enum class fieldType { ijk, ij, ik, jk, i, j, k };

enum class op {
  multiply,
  plus,
  minus,
  assign,
  divide,
  equal,
  notEqual,
  greater,
  less,
  greaterEqual,
  lessEqual,
  logicalAnd,
  locigalOr,
  logicalNot
};
enum class accessType { r, rw };
enum class hOffsetType { withOffset, noOffset };

// \brief Short syntax to build an IIR in a consistent state
//
// All return values should be used only once, except the variable creators (`field`, `localvar`).
// After creating the whole IIR, the stencil instantiation can be creating by calling build. The
// builder must not be used after calling build.
class IIRBuilder {
protected:
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
  IIRBuilder() : si_(std::make_shared<iir::StencilInstantiation>()) {}

  LocalVar localvar(std::string const& name, BuiltinTypeID = BuiltinTypeID::Float);

  std::shared_ptr<iir::Expr> reduceOverNeighborExpr(op operation, std::shared_ptr<iir::Expr>&& rhs,
                                                    std::shared_ptr<iir::Expr>&& init,
                                                    ast::Expr::LocationType rhs_location);

  std::shared_ptr<iir::Expr> binaryExpr(std::shared_ptr<iir::Expr>&& lhs,
                                        std::shared_ptr<iir::Expr>&& rhs, op operation);

  std::shared_ptr<iir::Expr> assignExpr(std::shared_ptr<iir::Expr>&& lhs,
                                        std::shared_ptr<iir::Expr>&& rhs,
                                        op operation = op::assign);

  std::shared_ptr<iir::Expr> unaryExpr(std::shared_ptr<iir::Expr>&& expr, op operation);

  std::shared_ptr<iir::Expr> conditionalExpr(std::shared_ptr<iir::Expr>&& cond,
                                             std::shared_ptr<iir::Expr>&& caseThen,
                                             std::shared_ptr<iir::Expr>&& caseElse);

  template <typename T>
  std::shared_ptr<iir::Expr> lit(T&& v) {
    DAWN_ASSERT(si_);
    auto v_str = std::to_string(std::forward<T>(v));
    int acc = si_->getMetaData().insertAccessOfType(iir::FieldAccessType::FAT_Literal, v_str);
    auto expr = std::make_shared<iir::LiteralAccessExpr>(
        v_str,
        sir::Value::typeToBuiltinTypeID(sir::Value::TypeInfo<typename std::decay<T>::type>::Type));
    expr->setID(-si_->nextUID());
    expr->template getData<IIRAccessExprData>().AccessID = std::make_optional(acc);
    return expr;
  }

  std::shared_ptr<iir::Expr> at(Field const& field, accessType access, ast::Offsets const& offset);

  std::shared_ptr<iir::Expr> at(LocalVar const& var);

  std::shared_ptr<iir::Stmt> stmt(std::shared_ptr<iir::Expr>&& expr);

  template <typename... Stmts>
  std::shared_ptr<iir::Stmt> block(Stmts&&... stmts) {
    DAWN_ASSERT(si_);
    auto stmt = iir::makeBlockStmt(std::vector<std::shared_ptr<iir::Stmt>>{std::move(stmts)...});
    return stmt;
  }

  std::shared_ptr<iir::Stmt> ifStmt(std::shared_ptr<iir::Expr>&& cond,
                                    std::shared_ptr<iir::Stmt>&& caseThen,
                                    std::shared_ptr<iir::Stmt>&& caseElse = {nullptr});

  std::shared_ptr<iir::Stmt> declareVar(LocalVar& var_id);

  template <typename... Stmts>
  std::unique_ptr<iir::DoMethod> vregion(sir::Interval::LevelKind s, sir::Interval::LevelKind e,
                                         Stmts&&... stmts) {
    DAWN_ASSERT(si_);
    auto ret = std::make_unique<iir::DoMethod>(iir::Interval(s, e), si_->getMetaData());
    ret->setID(si_->nextUID());
    [[maybe_unused]] int x[] = {(DAWN_ASSERT(stmts), ret->insertChild(std::move(stmts)), 0)...};
    computeAccesses(si_.get(), ret->getChildren());
    ret->updateLevel();
    return ret;
  }

  // specialized builder for the stage that accepts a location type
  template <typename... DoMethods>
  std::unique_ptr<iir::Stage> stage(ast::Expr::LocationType type, DoMethods&&... do_methods) {
    DAWN_ASSERT(si_);
    auto ret = std::make_unique<iir::Stage>(si_->getMetaData(), si_->nextUID());
    ret->setLocationType(type);
    int x[] = {(ret->insertChild(std::forward<DoMethods>(do_methods)), 0)...};
    (void)x;
    return ret;
  }

  // default builder for the stage that assumes stages are over cells
  template <typename... DoMethods>
  std::unique_ptr<iir::Stage> stage(DoMethods&&... do_methods) {
    DAWN_ASSERT(si_);
    auto ret = std::make_unique<iir::Stage>(si_->getMetaData(), si_->nextUID());
    int x[] = {(ret->insertChild(std::forward<DoMethods>(do_methods)), 0)...};
    (void)x;
    return ret;
  }

  template <typename... Stages>
  std::unique_ptr<iir::MultiStage> multistage(iir::LoopOrderKind loop_kind, Stages&&... stages) {
    DAWN_ASSERT(si_);
    auto ret = std::make_unique<iir::MultiStage>(si_->getMetaData(), loop_kind);
    ret->setID(si_->nextUID());
    int x[] = {(ret->insertChild(std::forward<Stages>(stages)), 0)...};
    (void)x;
    return ret;
  }

  template <typename... MultiStages>
  std::unique_ptr<iir::Stencil> stencil(MultiStages&&... multistages) {
    DAWN_ASSERT(si_);
    auto ret = std::make_unique<iir::Stencil>(si_->getMetaData(), sir::Attr{}, si_->nextUID());
    int x[] = {(ret->insertChild(std::forward<MultiStages>(multistages)), 0)...};
    (void)x;
    return ret;
  }

  // generates the final instantiation context
  dawn::codegen::stencilInstantiationContext build(std::string const& name,
                                                   std::unique_ptr<iir::Stencil> stencil);

protected:
  std::shared_ptr<iir::StencilInstantiation> si_;
};

class UnstructuredIIRBuilder : public IIRBuilder {
public:
  using IIRBuilder::at;
  std::shared_ptr<iir::Expr> at(Field const& field, accessType access, hOffsetType hOffset,
                                int vOffset);
  std::shared_ptr<iir::Expr> at(Field const& field, hOffsetType hOffset, int vOffset);
  std::shared_ptr<iir::Expr> at(Field const& field, accessType access = accessType::r);

  Field field(std::string const& name, ast::Expr::LocationType location);
};

class CartesianIIRBuilder : public IIRBuilder {
public:
  using IIRBuilder::at;
  std::shared_ptr<iir::Expr> at(Field const& field, accessType access, Array3i const& offset);
  std::shared_ptr<iir::Expr> at(Field const& field, Array3i const& offset);
  std::shared_ptr<iir::Expr> at(Field const& field, accessType access = accessType::r);

  Field field(std::string const& name, fieldType ft = fieldType::ijk);
};
} // namespace iir
} // namespace dawn

#endif
