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

#include "dawn/AST/AST.h"
#include "dawn/AST/ASTExpr.h"
#include "dawn/AST/ASTUtil.h"
#include "dawn/AST/ASTVisitor.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/Format.h"

namespace dawn {
namespace ast {

//===------------------------------------------------------------------------------------------===//
//     BlockStmt
//===------------------------------------------------------------------------------------------===//

template <typename DataTraits>
BlockStmt<DataTraits>::BlockStmt(SourceLocation loc) : Stmt<DataTraits>(SK_BlockStmt, loc) {}

template <typename DataTraits>
BlockStmt<DataTraits>::BlockStmt(const std::vector<std::shared_ptr<Stmt<DataTraits>>>& statements,
                                 SourceLocation loc)
    : Stmt<DataTraits>(SK_BlockStmt, loc), statements_(statements) {}

template <typename DataTraits>
BlockStmt<DataTraits>::BlockStmt(const BlockStmt<DataTraits>& stmt)
    : DataTraits::StmtData(), DataTraits::BlockStmt(stmt), Stmt<DataTraits>(
                                                               SK_BlockStmt,
                                                               stmt.getSourceLocation()) {
  for(auto s : stmt.getStatements())
    statements_.push_back(s->clone());
}

template <typename DataTraits>
BlockStmt<DataTraits>& BlockStmt<DataTraits>::operator=(BlockStmt<DataTraits> const& stmt) {
  DataTraits::BlockStmt::operator=(stmt);
  assign(stmt);
  statements_ = std::move(stmt.statements_);
  return *this;
}

template <typename DataTraits>
BlockStmt<DataTraits>::~BlockStmt() {}

template <typename DataTraits>
std::shared_ptr<Stmt<DataTraits>> BlockStmt<DataTraits>::clone() const {
  return std::make_shared<BlockStmt<DataTraits>>(*this);
}

template <typename DataTraits>
bool BlockStmt<DataTraits>::equals(const Stmt<DataTraits>* other) const {
  const BlockStmt<DataTraits>* otherPtr = dyn_cast<BlockStmt<DataTraits>>(other);
  return otherPtr && Stmt<DataTraits>::equals(other) &&
         otherPtr->statements_.size() == statements_.size() &&
         std::equal(statements_.begin(), statements_.end(), otherPtr->statements_.begin(),
                    [](const std::shared_ptr<Stmt<DataTraits>>& a,
                       const std::shared_ptr<Stmt<DataTraits>>& b) { return a->equals(b.get()); });
}

template <typename DataTraits>
void BlockStmt<DataTraits>::replaceChildren(std::shared_ptr<Stmt<DataTraits>> const& oldStmt,
                                            std::shared_ptr<Stmt<DataTraits>> const& newStmt) {
  bool success = ASTHelper::replaceOperands(oldStmt, newStmt, statements_);
  DAWN_ASSERT_MSG((success), ("Expression not found"));
}

//===------------------------------------------------------------------------------------------===//
//     ExprStmt
//===------------------------------------------------------------------------------------------===//

template <typename DataTraits>
ExprStmt<DataTraits>::ExprStmt(const std::shared_ptr<Expr<DataTraits>>& expr, SourceLocation loc)
    : Stmt<DataTraits>(SK_ExprStmt, loc), expr_(expr) {}

template <typename DataTraits>
ExprStmt<DataTraits>::ExprStmt(const ExprStmt<DataTraits>& stmt)
    : DataTraits::StmtData(), DataTraits::ExprStmt(stmt), Stmt<DataTraits>(
                                                              SK_ExprStmt,
                                                              stmt.getSourceLocation()),
      expr_(stmt.getExpr()->clone()) {}

template <typename DataTraits>
ExprStmt<DataTraits>& ExprStmt<DataTraits>::operator=(ExprStmt<DataTraits> stmt) {
  DataTraits::ExprStmt::operator=(stmt);
  assign(stmt);
  expr_ = stmt.getExpr();
  return *this;
}

template <typename DataTraits>
ExprStmt<DataTraits>::~ExprStmt() {}

template <typename DataTraits>
std::shared_ptr<Stmt<DataTraits>> ExprStmt<DataTraits>::clone() const {
  return std::make_shared<ExprStmt<DataTraits>>(*this);
}

template <typename DataTraits>
bool ExprStmt<DataTraits>::equals(const Stmt<DataTraits>* other) const {
  const ExprStmt<DataTraits>* otherPtr = dyn_cast<ExprStmt<DataTraits>>(other);
  return otherPtr && Stmt<DataTraits>::equals(other) && expr_->equals(otherPtr->expr_.get());
}

template <typename DataTraits>
void ExprStmt<DataTraits>::replaceChildren(std::shared_ptr<Expr<DataTraits>> const& oldExpr,
                                           std::shared_ptr<Expr<DataTraits>> const& newExpr) {
  DAWN_ASSERT_MSG((oldExpr == expr_ && oldExpr && newExpr), ("Expression not found"));
  expr_ = newExpr;
}

//===------------------------------------------------------------------------------------------===//
//     ReturnStmt
//===------------------------------------------------------------------------------------------===//

template <typename DataTraits>
ReturnStmt<DataTraits>::ReturnStmt(const std::shared_ptr<Expr<DataTraits>>& expr,
                                   SourceLocation loc)
    : Stmt<DataTraits>(SK_ReturnStmt, loc), expr_(expr) {}

template <typename DataTraits>
ReturnStmt<DataTraits>::ReturnStmt(const ReturnStmt<DataTraits>& stmt)
    : DataTraits::StmtData(), DataTraits::ReturnStmt(stmt), Stmt<DataTraits>(
                                                                SK_ReturnStmt,
                                                                stmt.getSourceLocation()),
      expr_(stmt.getExpr()->clone()) {}

template <typename DataTraits>
ReturnStmt<DataTraits>& ReturnStmt<DataTraits>::operator=(ReturnStmt<DataTraits> stmt) {
  DataTraits::ReturnStmt::operator=(stmt);
  assign(stmt);
  expr_ = stmt.getExpr();
  return *this;
}

template <typename DataTraits>
ReturnStmt<DataTraits>::~ReturnStmt() {}

template <typename DataTraits>
std::shared_ptr<Stmt<DataTraits>> ReturnStmt<DataTraits>::clone() const {
  return std::make_shared<ReturnStmt<DataTraits>>(*this);
}

template <typename DataTraits>
bool ReturnStmt<DataTraits>::equals(const Stmt<DataTraits>* other) const {
  const ReturnStmt<DataTraits>* otherPtr = dyn_cast<ReturnStmt<DataTraits>>(other);
  return otherPtr && Stmt<DataTraits>::equals(other) && expr_->equals(otherPtr->expr_.get());
}

template <typename DataTraits>
void ReturnStmt<DataTraits>::replaceChildren(std::shared_ptr<Expr<DataTraits>> const& oldExpr,
                                             std::shared_ptr<Expr<DataTraits>> const& newExpr) {
  DAWN_ASSERT_MSG((oldExpr == expr_), ("Expression not found"));
  expr_ = newExpr;
}

//===------------------------------------------------------------------------------------------===//
//     VarDeclStmt
//===------------------------------------------------------------------------------------------===//

template <typename DataTraits>
VarDeclStmt<DataTraits>::VarDeclStmt(const Type& type, const std::string& name, int dimension,
                                     const char* op,
                                     std::vector<std::shared_ptr<Expr<DataTraits>>> initList,
                                     SourceLocation loc)
    : Stmt<DataTraits>(SK_VarDeclStmt, loc), type_(type), name_(name), dimension_(dimension),
      op_(op), initList_(std::move(initList)) {}

template <typename DataTraits>
VarDeclStmt<DataTraits>::VarDeclStmt(const VarDeclStmt<DataTraits>& stmt)
    : DataTraits::StmtData(), DataTraits::VarDeclStmt(stmt), Stmt<DataTraits>(
                                                                 SK_VarDeclStmt,
                                                                 stmt.getSourceLocation()),
      type_(stmt.getType()), name_(stmt.getName()), dimension_(stmt.getDimension()),
      op_(stmt.getOp()) {
  for(const auto& expr : stmt.getInitList())
    initList_.push_back(expr->clone());
}

template <typename DataTraits>
VarDeclStmt<DataTraits>& VarDeclStmt<DataTraits>::operator=(VarDeclStmt<DataTraits> stmt) {
  DataTraits::VarDeclStmt::operator=(stmt);
  assign(stmt);
  type_ = std::move(stmt.getType());
  name_ = std::move(stmt.getName());
  dimension_ = stmt.getDimension();
  op_ = stmt.getOp();
  initList_ = std::move(stmt.getInitList());
  return *this;
}

template <typename DataTraits>
VarDeclStmt<DataTraits>::~VarDeclStmt() {}

template <typename DataTraits>
std::shared_ptr<Stmt<DataTraits>> VarDeclStmt<DataTraits>::clone() const {
  return std::make_shared<VarDeclStmt<DataTraits>>(*this);
}

template <typename DataTraits>
bool VarDeclStmt<DataTraits>::equals(const Stmt<DataTraits>* other) const {
  const VarDeclStmt<DataTraits>* otherPtr = dyn_cast<VarDeclStmt<DataTraits>>(other);
  return otherPtr && Stmt<DataTraits>::equals(other) && type_ == otherPtr->type_ &&
         name_ == otherPtr->name_ && dimension_ == otherPtr->dimension_ && op_ == otherPtr->op_ &&
         initList_.size() == otherPtr->initList_.size() &&
         std::equal(initList_.begin(), initList_.end(), otherPtr->initList_.begin(),
                    [](const std::shared_ptr<Expr<DataTraits>>& a,
                       const std::shared_ptr<Expr<DataTraits>>& b) { return a->equals(b.get()); });
}

template <typename DataTraits>
void VarDeclStmt<DataTraits>::replaceChildren(std::shared_ptr<Expr<DataTraits>> const& oldExpr,
                                              std::shared_ptr<Expr<DataTraits>> const& newExpr) {
  bool success = ASTHelper::replaceOperands(oldExpr, newExpr, initList_);
  DAWN_ASSERT_MSG((success), ("Expression not found"));
}

//===------------------------------------------------------------------------------------------===//
//     StencilCall
//===------------------------------------------------------------------------------------------===//

inline std::shared_ptr<StencilCall> StencilCall::clone() const {
  auto call = std::make_shared<StencilCall>(Callee, Loc);
  call->Args = Args;
  return call;
}

inline bool StencilCall::operator==(const StencilCall& rhs) const {
  return bool(this->comparison(rhs));
}

inline CompareResult StencilCall::comparison(const StencilCall& rhs) const {
  std::string output;
  if(Callee != rhs.Callee) {
    output += dawn::format("[StencilCall mismatch] Callees do not match\n"
                           "  Actual:\n"
                           "    %s\n"
                           "  Expected:\n"
                           "    %s",
                           Callee, rhs.Callee);
    return CompareResult{output, false};
  }
  for(int i = 0; i < Args.size(); ++i) {
    if(Args[i] != rhs.Args[i]) {
      output += "[StencilCall mismatch] Arguments do not match\n";
      output += dawn::format("Names do not match\n"
                             "  Actual:\n"
                             "    %s\n"
                             "  Expected:\n"
                             "    %s",
                             Args[i], rhs.Args[i]);
      return CompareResult{output, false};
    }
  }
  return CompareResult{output, true};
}

//===------------------------------------------------------------------------------------------===//
//     StencilCallDeclStmt
//===------------------------------------------------------------------------------------------===//

template <typename DataTraits>
StencilCallDeclStmt<DataTraits>::StencilCallDeclStmt(
    const std::shared_ptr<StencilCall>& stencilCall, SourceLocation loc)
    : Stmt<DataTraits>(SK_StencilCallDeclStmt, loc), stencilCall_(stencilCall) {}

template <typename DataTraits>
StencilCallDeclStmt<DataTraits>::StencilCallDeclStmt(const StencilCallDeclStmt<DataTraits>& stmt)
    : DataTraits::StmtData(), DataTraits::StencilCallDeclStmt(stmt), Stmt<DataTraits>(
                                                                         SK_StencilCallDeclStmt,
                                                                         stmt.getSourceLocation()),
      stencilCall_(stmt.getStencilCall()->clone()) {}

template <typename DataTraits>
StencilCallDeclStmt<DataTraits>& StencilCallDeclStmt<DataTraits>::
operator=(StencilCallDeclStmt<DataTraits> stmt) {
  DataTraits::StencilCallDeclStmt::operator=(stmt);
  assign(stmt);
  stencilCall_ = std::move(stmt.getStencilCall());
  return *this;
}

template <typename DataTraits>
StencilCallDeclStmt<DataTraits>::~StencilCallDeclStmt() {}

template <typename DataTraits>
std::shared_ptr<Stmt<DataTraits>> StencilCallDeclStmt<DataTraits>::clone() const {
  return std::make_shared<StencilCallDeclStmt<DataTraits>>(*this);
}

template <typename DataTraits>
bool StencilCallDeclStmt<DataTraits>::equals(const Stmt<DataTraits>* other) const {
  const StencilCallDeclStmt<DataTraits>* otherPtr =
      dyn_cast<StencilCallDeclStmt<DataTraits>>(other);
  if(otherPtr) {
    if(stencilCall_) {
      auto res = stencilCall_->comparison(*otherPtr->stencilCall_);
      if(!bool(res)) {
        return false;
      }
    }
    return Stmt<DataTraits>::equals(other);
  }
  return false;
}

//===------------------------------------------------------------------------------------------===//
//     BoundaryConditionDeclStmt
//===------------------------------------------------------------------------------------------===//

template <typename DataTraits>
BoundaryConditionDeclStmt<DataTraits>::BoundaryConditionDeclStmt(const std::string& callee,
                                                                 SourceLocation loc)
    : Stmt<DataTraits>(SK_BoundaryConditionDeclStmt, loc), functor_(callee) {}

template <typename DataTraits>
BoundaryConditionDeclStmt<DataTraits>::BoundaryConditionDeclStmt(
    const BoundaryConditionDeclStmt<DataTraits>& stmt)
    : DataTraits::StmtData(), DataTraits::BoundaryConditionDeclStmt(stmt),
      Stmt<DataTraits>(SK_BoundaryConditionDeclStmt, stmt.getSourceLocation()),
      functor_(stmt.functor_), fields_(stmt.fields_) {}

template <typename DataTraits>
BoundaryConditionDeclStmt<DataTraits>& BoundaryConditionDeclStmt<DataTraits>::
operator=(BoundaryConditionDeclStmt<DataTraits> stmt) {
  DataTraits::BoundaryConditionDeclStmt::operator=(stmt);
  assign(stmt);
  functor_ = std::move(stmt.functor_);
  fields_ = std::move(stmt.fields_);
  return *this;
}

template <typename DataTraits>
BoundaryConditionDeclStmt<DataTraits>::~BoundaryConditionDeclStmt() {}

template <typename DataTraits>
std::shared_ptr<Stmt<DataTraits>> BoundaryConditionDeclStmt<DataTraits>::clone() const {
  return std::make_shared<BoundaryConditionDeclStmt<DataTraits>>(*this);
}

template <typename DataTraits>
bool BoundaryConditionDeclStmt<DataTraits>::equals(const Stmt<DataTraits>* other) const {
  const BoundaryConditionDeclStmt<DataTraits>* otherPtr =
      dyn_cast<BoundaryConditionDeclStmt<DataTraits>>(other);
  return otherPtr && Stmt<DataTraits>::equals(other) && functor_ == otherPtr->functor_ &&
         fields_.size() == otherPtr->fields_.size() &&
         std::equal(fields_.begin(), fields_.end(), otherPtr->fields_.begin(),
                    [](const std::string& a, const std::string& b) { return a == b; });
}

//===------------------------------------------------------------------------------------------===//
//     IfStmt
//===------------------------------------------------------------------------------------------===//

template <typename DataTraits>
IfStmt<DataTraits>::IfStmt(const std::shared_ptr<Stmt<DataTraits>>& condStmt,
                           const std::shared_ptr<Stmt<DataTraits>>& thenStmt,
                           const std::shared_ptr<Stmt<DataTraits>>& elseStmt, SourceLocation loc)
    : Stmt<DataTraits>(SK_IfStmt, loc), subStmts_{condStmt, thenStmt, elseStmt} {}

template <typename DataTraits>
IfStmt<DataTraits>::IfStmt(const IfStmt<DataTraits>& stmt)
    : DataTraits::StmtData(), DataTraits::IfStmt(stmt), Stmt<DataTraits>(SK_IfStmt,
                                                                         stmt.getSourceLocation()),
      subStmts_{stmt.getCondStmt()->clone(), stmt.getThenStmt()->clone(),
                stmt.hasElse() ? stmt.getElseStmt()->clone() : nullptr} {}

template <typename DataTraits>
IfStmt<DataTraits>& IfStmt<DataTraits>::operator=(IfStmt<DataTraits> stmt) {
  DataTraits::IfStmt::operator=(stmt);
  assign(stmt);
  subStmts_[OK_Cond] = std::move(stmt.getCondStmt());
  subStmts_[OK_Then] = std::move(stmt.getThenStmt());
  subStmts_[OK_Else] = std::move(stmt.getElseStmt());
  return *this;
}

template <typename DataTraits>
IfStmt<DataTraits>::~IfStmt() {}

template <typename DataTraits>
std::shared_ptr<Stmt<DataTraits>> IfStmt<DataTraits>::clone() const {
  return std::make_shared<IfStmt<DataTraits>>(*this);
}

template <typename DataTraits>
bool IfStmt<DataTraits>::equals(const Stmt<DataTraits>* other) const {
  const IfStmt<DataTraits>* otherPtr = dyn_cast<IfStmt<DataTraits>>(other);
  bool sameElse;
  if(hasElse() && otherPtr->hasElse())
    sameElse = subStmts_[OK_Else]->equals(otherPtr->subStmts_[OK_Else].get());
  else
    sameElse = !(hasElse() ^ otherPtr->hasElse());
  return otherPtr && Stmt<DataTraits>::equals(other) &&
         subStmts_[OK_Cond]->equals(otherPtr->subStmts_[OK_Cond].get()) &&
         subStmts_[OK_Then]->equals(otherPtr->subStmts_[OK_Then].get()) && sameElse;
}

template <typename DataTraits>
void IfStmt<DataTraits>::replaceChildren(std::shared_ptr<Stmt<DataTraits>> const& oldStmt,
                                         std::shared_ptr<Stmt<DataTraits>> const& newStmt) {
  if(hasElse()) {
    for(std::shared_ptr<Stmt<DataTraits>>& stmt : subStmts_) {
      if(stmt == oldStmt)
        stmt = newStmt;
      return;
    }
  } else {
    DAWN_ASSERT(oldStmt == subStmts_[0]);
    subStmts_[0] = newStmt;
    return;
  }
  DAWN_ASSERT_MSG((false), ("Expression not found"));
}
} // namespace ast
} // namespace dawn
