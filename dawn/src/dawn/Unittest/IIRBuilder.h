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
#include "dawn/IIR/AccessComputation.h"
#include "dawn/IIR/Extents.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/Stage.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/SIR.h"

namespace dawn {
namespace iir {

enum class FieldType { ijk, ij, ik, jk, i, j, k };

enum class Op {
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
enum class AccessType { r, rw };
enum class HOffsetType { withOffset, noOffset };

// \brief Short syntax to build an IIR in a consistent state
//
// All return values should be used only once, except the variable creators (`field`, `localvar`).
// After creating the whole IIR, the stencil instantiation can be creating by calling build. The
// builder must not be used after calling build.
class IIRBuilder {
private:
  static std::string toStr(Op operation, std::vector<Op> const& valid_ops) {
    DAWN_ASSERT(std::find(valid_ops.begin(), valid_ops.end(), operation) != valid_ops.end());
    switch(operation) {
    case Op::plus:
      return "+";
    case Op::minus:
      return "-";
    case Op::multiply:
      return "*";
    case Op::assign:
      return "";
    case Op::divide:
      return "/";
    case Op::equal:
      return "==";
    case Op::notEqual:
      return "!=";
    case Op::greater:
      return ">";
    case Op::less:
      return "<";
    case Op::greaterEqual:
      return ">=";
    case Op::lessEqual:
      return "<=";
    case Op::logicalAnd:
      return "&&";
    case Op::locigalOr:
      return "||";
    case Op::logicalNot:
      return "!";
    }
    dawn_unreachable("Unreachable");
  }

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
  IIRBuilder(const ast::GridType gridType)
      : si_(std::make_shared<iir::StencilInstantiation>(gridType)) {}

  LocalVar localvar(std::string const& name, BuiltinTypeID = BuiltinTypeID::Float,
                    std::vector<std::shared_ptr<iir::Expr>>&& initList = {});

  template <class TWeight>
  std::shared_ptr<iir::Expr>
  reduceOverNeighborExpr(Op operation, std::shared_ptr<iir::Expr>&& rhs,
                         std::shared_ptr<iir::Expr>&& init, ast::LocationType lhs_location,
                         ast::LocationType rhs_location, const std::vector<TWeight>&& weights) {
    static_assert(std::is_arithmetic<TWeight>::value, "weights need to be of arithmetic type!\n");

    std::vector<sir::Value> vWeights;
    for(const auto& it : weights) {
      vWeights.push_back(sir::Value(it));
    }

    auto expr = std::make_shared<iir::ReductionOverNeighborExpr>(
        toStr(operation, {Op::multiply, Op::plus, Op::minus, Op::assign, Op::divide}),
        std::move(rhs), std::move(init), vWeights, lhs_location, rhs_location);
    expr->setID(si_->nextUID());

    return expr;
  }

  std::shared_ptr<iir::Expr> reduceOverNeighborExpr(Op operation, std::shared_ptr<iir::Expr>&& rhs,
                                                    std::shared_ptr<iir::Expr>&& init,
                                                    ast::LocationType lhs_location,
                                                    ast::LocationType rhs_location);

  std::shared_ptr<iir::Expr> binaryExpr(std::shared_ptr<iir::Expr>&& lhs,
                                        std::shared_ptr<iir::Expr>&& rhs, Op operation = Op::plus);

  std::shared_ptr<iir::Expr> assignExpr(std::shared_ptr<iir::Expr>&& lhs,
                                        std::shared_ptr<iir::Expr>&& rhs,
                                        Op operation = Op::assign);

  std::shared_ptr<iir::Expr> unaryExpr(std::shared_ptr<iir::Expr>&& expr, Op operation);

  std::shared_ptr<iir::Expr> conditionalExpr(std::shared_ptr<iir::Expr>&& cond,
                                             std::shared_ptr<iir::Expr>&& caseThen,
                                             std::shared_ptr<iir::Expr>&& caseElse);

  template <typename T>
  std::shared_ptr<iir::Expr> lit(T&& v) {
    DAWN_ASSERT(si_);
    auto v_str = std::to_string(std::forward<T>(v));
    int acc = si_->getMetaData().insertAccessOfType(iir::FieldAccessType::Literal, v_str);
    auto expr = std::make_shared<iir::LiteralAccessExpr>(
        v_str,
        sir::Value::typeToBuiltinTypeID(sir::Value::TypeInfo<typename std::decay<T>::type>::Type));
    expr->setID(-si_->nextUID());
    expr->template getData<IIRAccessExprData>().AccessID = std::make_optional(acc);
    return expr;
  }

  std::shared_ptr<iir::Expr> at(Field const& field, AccessType access, ast::Offsets const& offset);

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
  std::unique_ptr<iir::DoMethod> doMethod(iir::Interval interval, Stmts&&... stmts) {
    DAWN_ASSERT(si_);
    auto ret = std::make_unique<iir::DoMethod>(interval, si_->getMetaData());
    ret->setID(si_->nextUID());
    [[maybe_unused]] int x[] = {
        (DAWN_ASSERT(stmts), ret->getAST().push_back(std::move(stmts)), 0)...};
    computeAccesses(si_.get(), ret->getAST().getStatements());
    ret->updateLevel();
    return ret;
  }

  template <typename... Stmts>
  std::unique_ptr<iir::DoMethod> doMethod(sir::Interval::LevelKind s, sir::Interval::LevelKind e,
                                          Stmts&&... stmts) {
    return doMethod(iir::Interval(s, e), stmts...);
  }

  template <typename... Stmts>
  std::unique_ptr<iir::DoMethod> doMethod(sir::Interval::LevelKind s, sir::Interval::LevelKind e,
                                          int offsetLow, int offsetHigh, Stmts&&... stmts) {
    DAWN_ASSERT(si_);
    auto ret = std::make_unique<iir::DoMethod>(iir::Interval(s, e, offsetLow, offsetHigh),
                                               si_->getMetaData());
    ret->setID(si_->nextUID());
    [[maybe_unused]] int x[] = {
        (DAWN_ASSERT(stmts), ret->getAST().push_back(std::move(stmts)), 0)...};
    computeAccesses(si_.get(), ret->getAST().getStatements());
    ret->updateLevel();
    return ret;
  }

  // specialized builder for the stage that accepts a location type
  template <typename... DoMethods>
  std::unique_ptr<iir::Stage> stage(ast::LocationType type, DoMethods&&... do_methods) {
    DAWN_ASSERT(si_);
    auto ret = std::make_unique<iir::Stage>(si_->getMetaData(), si_->nextUID());
    ret->setLocationType(type);
    [[maybe_unused]] int x[] = {(ret->insertChild(std::forward<DoMethods>(do_methods)), 0)...};
    (void)x;
    return ret;
  }

  // specialized builder for the stage that accepts a global index
  template <typename... DoMethods>
  std::unique_ptr<iir::Stage> stage(int direction, Interval interval, DoMethods&&... do_methods) {
    DAWN_ASSERT(si_);
    auto ret = std::make_unique<iir::Stage>(si_->getMetaData(), si_->nextUID());
    iir::Stage::IterationSpace iterationSpace;
    iterationSpace[direction] = interval;
    ret->setIterationSpace(iterationSpace);
    [[maybe_unused]] int x[] = {(ret->insertChild(std::forward<DoMethods>(do_methods)), 0)...};
    (void)x;
    return ret;
  }

  // specialized builder for the stage that accepts a global index
  template <typename... DoMethods>
  std::unique_ptr<iir::Stage> stage(Interval intervalI, Interval intervalJ,
                                    DoMethods&&... do_methods) {
    DAWN_ASSERT(si_);
    auto ret = std::make_unique<iir::Stage>(si_->getMetaData(), si_->nextUID());
    iir::Stage::IterationSpace iterationSpace;
    iterationSpace[0] = intervalI;
    iterationSpace[1] = intervalJ;
    ret->setIterationSpace(iterationSpace);
    [[maybe_unused]] int x[] = {(ret->insertChild(std::forward<DoMethods>(do_methods)), 0)...};
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
  UnstructuredIIRBuilder() : IIRBuilder(ast::GridType::Unstructured) {}
  using IIRBuilder::at;
  std::shared_ptr<iir::Expr> at(Field const& field, AccessType access, HOffsetType hOffset,
                                int vOffset);
  std::shared_ptr<iir::Expr> at(Field const& field, HOffsetType hOffset, int vOffset);
  std::shared_ptr<iir::Expr> at(Field const& field, AccessType access = AccessType::r);

  Field field(std::string const& name, ast::LocationType denseLocation);
  Field field(std::string const& name, ast::NeighborChain sparseChain);
};

class CartesianIIRBuilder : public IIRBuilder {
public:
  CartesianIIRBuilder() : IIRBuilder(ast::GridType::Cartesian) {}
  using IIRBuilder::at;
  std::shared_ptr<iir::Expr> at(Field const& field, AccessType access, Array3i const& offset);
  std::shared_ptr<iir::Expr> at(Field const& field, Array3i const& offset);
  std::shared_ptr<iir::Expr> at(Field const& field, AccessType access = AccessType::r);

  Field field(std::string const& name, FieldType ft = FieldType::ijk);
  Field tmpField(std::string const& name, FieldType ft = FieldType::ijk);
};
} // namespace iir
} // namespace dawn

#endif
