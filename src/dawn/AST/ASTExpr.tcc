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

#include "dawn/AST/ASTUtil.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/StringRef.h"

namespace dawn {
namespace ast {
//===------------------------------------------------------------------------------------------===//
//     UnaryOperator
//===------------------------------------------------------------------------------------------===//

template <typename DataTraits>
UnaryOperator<DataTraits>::UnaryOperator(const std::shared_ptr<Expr<DataTraits>>& operand,
                                         std::string op, SourceLocation loc)
    : Expr<DataTraits>(EK_UnaryOperator, loc), operand_(operand), op_(std::move(op)) {}

template <typename DataTraits>
UnaryOperator<DataTraits>::UnaryOperator(const UnaryOperator<DataTraits>& expr)
    : Expr<DataTraits>(EK_UnaryOperator, expr.getSourceLocation()),
      operand_(expr.getOperand()->clone()), op_(expr.getOp()) {}

template <typename DataTraits>
UnaryOperator<DataTraits>& UnaryOperator<DataTraits>::operator=(UnaryOperator<DataTraits> expr) {
  assign(expr);
  operand_ = expr.getOperand();
  op_ = expr.getOp();
  return *this;
}

template <typename DataTraits>
UnaryOperator<DataTraits>::~UnaryOperator() {}

template <typename DataTraits>
std::shared_ptr<Expr<DataTraits>> UnaryOperator<DataTraits>::clone() const {
  return std::make_shared<UnaryOperator<DataTraits>>(*this);
}

template <typename DataTraits>
bool UnaryOperator<DataTraits>::equals(const Expr<DataTraits>* other) const {
  const UnaryOperator<DataTraits>* otherPtr = dyn_cast<UnaryOperator<DataTraits>>(other);
  return otherPtr && Expr<DataTraits>::equals(other) &&
         operand_->equals(otherPtr->operand_.get()) && op_ == otherPtr->op_;
}

template <typename DataTraits>
void UnaryOperator<DataTraits>::replaceChildren(const std::shared_ptr<Expr<DataTraits>>& oldExpr,
                                                const std::shared_ptr<Expr<DataTraits>>& newExpr) {
  DAWN_ASSERT(oldExpr == operand_);
  operand_ = newExpr;
}

//===------------------------------------------------------------------------------------------===//
//     BinaryOperator
//===------------------------------------------------------------------------------------------===//

template <typename DataTraits>
BinaryOperator<DataTraits>::BinaryOperator(const std::shared_ptr<Expr<DataTraits>>& left,
                                           std::string op,
                                           const std::shared_ptr<Expr<DataTraits>>& right,
                                           SourceLocation loc)
    : Expr<DataTraits>(EK_BinaryOperator, loc), operands_{left, right}, op_(std::move(op)) {}

template <typename DataTraits>
BinaryOperator<DataTraits>::BinaryOperator(const BinaryOperator<DataTraits>& expr)
    : Expr<DataTraits>(EK_BinaryOperator, expr.getSourceLocation()),
      operands_{expr.getLeft()->clone(), expr.getRight()->clone()}, op_(expr.getOp()) {}

template <typename DataTraits>
BinaryOperator<DataTraits>& BinaryOperator<DataTraits>::operator=(BinaryOperator<DataTraits> expr) {
  assign(expr);
  operands_[OK_Left] = expr.getLeft();
  operands_[OK_Right] = expr.getRight();
  op_ = expr.getOp();
  return *this;
}

template <typename DataTraits>
BinaryOperator<DataTraits>::~BinaryOperator() {}

template <typename DataTraits>
std::shared_ptr<Expr<DataTraits>> BinaryOperator<DataTraits>::clone() const {
  return std::make_shared<BinaryOperator<DataTraits>>(*this);
}

template <typename DataTraits>
bool BinaryOperator<DataTraits>::equals(const Expr<DataTraits>* other) const {
  const BinaryOperator<DataTraits>* otherPtr = dyn_cast<BinaryOperator<DataTraits>>(other);
  return otherPtr && Expr<DataTraits>::equals(other) &&
         operands_[OK_Left]->equals(otherPtr->operands_[OK_Left].get()) &&
         operands_[OK_Right]->equals(otherPtr->operands_[OK_Right].get()) && op_ == otherPtr->op_;
}

template <typename DataTraits>
void BinaryOperator<DataTraits>::replaceChildren(const std::shared_ptr<Expr<DataTraits>>& oldExpr,
                                                 const std::shared_ptr<Expr<DataTraits>>& newExpr) {
  bool success = ASTHelper::replaceOperands(oldExpr, newExpr, operands_);
  DAWN_ASSERT_MSG((success), ("Expression not found"));
}

//===------------------------------------------------------------------------------------------===//
//     AssignmentExpr
//===------------------------------------------------------------------------------------------===//

template <typename DataTraits>
AssignmentExpr<DataTraits>::AssignmentExpr(const std::shared_ptr<Expr<DataTraits>>& left,
                                           const std::shared_ptr<Expr<DataTraits>>& right,
                                           std::string op, SourceLocation loc)
    : BinaryOperator<DataTraits>(left, std::move(op), right, loc) {
  kind_ = EK_AssignmentExpr;
}

template <typename DataTraits>
AssignmentExpr<DataTraits>::AssignmentExpr(const AssignmentExpr<DataTraits>& expr)
    : BinaryOperator<DataTraits>(expr.getLeft()->clone(), expr.getOp(), expr.getRight()->clone(),
                                 expr.getSourceLocation()) {
  kind_ = EK_AssignmentExpr;
}

template <typename DataTraits>
AssignmentExpr<DataTraits>& AssignmentExpr<DataTraits>::operator=(AssignmentExpr<DataTraits> expr) {
  assign(expr);
  operands_[OK_Left] = expr.getLeft();
  operands_[OK_Right] = expr.getRight();
  op_ = expr.getOp();
  return *this;
}

template <typename DataTraits>
AssignmentExpr<DataTraits>::~AssignmentExpr() {}

template <typename DataTraits>
std::shared_ptr<Expr<DataTraits>> AssignmentExpr<DataTraits>::clone() const {
  return std::make_shared<AssignmentExpr<DataTraits>>(*this);
}

template <typename DataTraits>
bool AssignmentExpr<DataTraits>::equals(const Expr<DataTraits>* other) const {
  const AssignmentExpr<DataTraits>* otherPtr = dyn_cast<AssignmentExpr<DataTraits>>(other);
  return otherPtr && Expr<DataTraits>::equals(other) &&
         operands_[OK_Left]->equals(otherPtr->operands_[OK_Left].get()) &&
         operands_[OK_Right]->equals(otherPtr->operands_[OK_Right].get()) && op_ == otherPtr->op_;
}

//===------------------------------------------------------------------------------------------===//
//     NOPExpr
//===------------------------------------------------------------------------------------------===//

template <typename DataTraits>
NOPExpr<DataTraits>::NOPExpr(SourceLocation loc) : Expr<DataTraits>(EK_NOPExpr, loc) {
  kind_ = EK_NOPExpr;
}

template <typename DataTraits>
NOPExpr<DataTraits>::NOPExpr(const NOPExpr<DataTraits>& expr)
    : Expr<DataTraits>(EK_NOPExpr, expr.getSourceLocation()) {
  kind_ = EK_NOPExpr;
}

template <typename DataTraits>
NOPExpr<DataTraits>& NOPExpr<DataTraits>::operator=(NOPExpr<DataTraits> expr) {
  assign(expr);
  return *this;
}

template <typename DataTraits>
NOPExpr<DataTraits>::~NOPExpr() {}

template <typename DataTraits>
std::shared_ptr<Expr<DataTraits>> NOPExpr<DataTraits>::clone() const {
  return std::make_shared<NOPExpr<DataTraits>>(*this);
}

template <typename DataTraits>
bool NOPExpr<DataTraits>::equals(const Expr<DataTraits>* other) const {
  return true;
}

//===------------------------------------------------------------------------------------------===//
//     TernaryOperator
//===------------------------------------------------------------------------------------------===//

template <typename DataTraits>
TernaryOperator<DataTraits>::TernaryOperator(const std::shared_ptr<Expr<DataTraits>>& cond,
                                             const std::shared_ptr<Expr<DataTraits>>& left,
                                             const std::shared_ptr<Expr<DataTraits>>& right,
                                             SourceLocation loc)
    : Expr<DataTraits>(EK_TernaryOperator, loc), operands_{cond, left, right} {}

template <typename DataTraits>
TernaryOperator<DataTraits>::TernaryOperator(const TernaryOperator<DataTraits>& expr)
    : Expr<DataTraits>(EK_TernaryOperator, expr.getSourceLocation()),
      operands_{expr.getCondition()->clone(), expr.getLeft()->clone(), expr.getRight()->clone()} {}

template <typename DataTraits>
TernaryOperator<DataTraits>& TernaryOperator<DataTraits>::
operator=(TernaryOperator<DataTraits> expr) {
  assign(expr);
  operands_[OK_Cond] = expr.getCondition();
  operands_[OK_Left] = expr.getLeft();
  operands_[OK_Right] = expr.getRight();
  return *this;
}

template <typename DataTraits>
TernaryOperator<DataTraits>::~TernaryOperator() {}

template <typename DataTraits>
std::shared_ptr<Expr<DataTraits>> TernaryOperator<DataTraits>::clone() const {
  return std::make_shared<TernaryOperator<DataTraits>>(*this);
}

template <typename DataTraits>
bool TernaryOperator<DataTraits>::equals(const Expr<DataTraits>* other) const {
  const TernaryOperator<DataTraits>* otherPtr = dyn_cast<TernaryOperator<DataTraits>>(other);
  return otherPtr && Expr<DataTraits>::equals(other) &&
         operands_[OK_Cond]->equals(otherPtr->operands_[OK_Cond].get()) &&
         operands_[OK_Left]->equals(otherPtr->operands_[OK_Left].get()) &&
         operands_[OK_Right]->equals(otherPtr->operands_[OK_Right].get());
}

template <typename DataTraits>
void TernaryOperator<DataTraits>::replaceChildren(
    const std::shared_ptr<Expr<DataTraits>>& oldExpr,
    const std::shared_ptr<Expr<DataTraits>>& newExpr) {
  bool success = ASTHelper::replaceOperands(oldExpr, newExpr, operands_);
  DAWN_ASSERT_MSG((success), ("Expression not found"));
}

//===------------------------------------------------------------------------------------------===//
//     FunCallExpr
//===------------------------------------------------------------------------------------------===//

template <typename DataTraits>
FunCallExpr<DataTraits>::FunCallExpr(const std::string& callee, SourceLocation loc)
    : Expr<DataTraits>(EK_FunCallExpr, loc), callee_(callee) {}

template <typename DataTraits>
FunCallExpr<DataTraits>::FunCallExpr(const FunCallExpr<DataTraits>& expr)
    : Expr<DataTraits>(EK_FunCallExpr, expr.getSourceLocation()), callee_(expr.getCallee()) {
  for(auto e : expr.getArguments())
    arguments_.push_back(e->clone());
}

template <typename DataTraits>
FunCallExpr<DataTraits>& FunCallExpr<DataTraits>::operator=(FunCallExpr<DataTraits> expr) {
  assign(expr);
  callee_ = std::move(expr.getCallee());
  arguments_ = std::move(expr.getArguments());
  return *this;
}

template <typename DataTraits>
FunCallExpr<DataTraits>::~FunCallExpr() {}

template <typename DataTraits>
std::shared_ptr<Expr<DataTraits>> FunCallExpr<DataTraits>::clone() const {
  return std::make_shared<FunCallExpr<DataTraits>>(*this);
}

template <typename DataTraits>
bool FunCallExpr<DataTraits>::equals(const Expr<DataTraits>* other) const {
  const FunCallExpr<DataTraits>* otherPtr = dyn_cast<FunCallExpr<DataTraits>>(other);
  return otherPtr && Expr<DataTraits>::equals(other) && callee_ == otherPtr->callee_ &&
         arguments_.size() == otherPtr->arguments_.size() &&
         std::equal(arguments_.begin(), arguments_.end(), otherPtr->arguments_.begin(),
                    [](const std::shared_ptr<Expr<DataTraits>>& a,
                       const std::shared_ptr<Expr<DataTraits>>& b) { return a->equals(b.get()); });
}

template <typename DataTraits>
void FunCallExpr<DataTraits>::insertArgument(const std::shared_ptr<Expr<DataTraits>>& expr) {
  arguments_.push_back(expr);
}

template <typename DataTraits>
void FunCallExpr<DataTraits>::replaceChildren(const std::shared_ptr<Expr<DataTraits>>& oldExpr,
                                              const std::shared_ptr<Expr<DataTraits>>& newExpr) {
  bool success = ASTHelper::replaceOperands(oldExpr, newExpr, arguments_);
  DAWN_ASSERT_MSG((success), ("Expression not found"));
}

//===------------------------------------------------------------------------------------------===//
//     StencilFunCallExpr
//===------------------------------------------------------------------------------------------===//

template <typename DataTraits>
StencilFunCallExpr<DataTraits>::StencilFunCallExpr(const std::string& callee, SourceLocation loc)
    : FunCallExpr<DataTraits>(callee, loc) {
  kind_ = EK_StencilFunCallExpr;
}

template <typename DataTraits>
StencilFunCallExpr<DataTraits>::StencilFunCallExpr(const StencilFunCallExpr<DataTraits>& expr)
    : FunCallExpr<DataTraits>(expr.getCallee(), expr.getSourceLocation()) {
  kind_ = EK_StencilFunCallExpr;
  for(auto e : expr.getArguments())
    arguments_.push_back(e->clone());
}

template <typename DataTraits>
StencilFunCallExpr<DataTraits>& StencilFunCallExpr<DataTraits>::
operator=(StencilFunCallExpr<DataTraits> expr) {
  assign(expr);
  callee_ = std::move(expr.getCallee());
  arguments_ = std::move(expr.getArguments());
  return *this;
}

template <typename DataTraits>
StencilFunCallExpr<DataTraits>::~StencilFunCallExpr() {}

template <typename DataTraits>
std::shared_ptr<Expr<DataTraits>> StencilFunCallExpr<DataTraits>::clone() const {
  return std::make_shared<StencilFunCallExpr<DataTraits>>(*this);
}

template <typename DataTraits>
bool StencilFunCallExpr<DataTraits>::equals(const Expr<DataTraits>* other) const {
  const StencilFunCallExpr<DataTraits>* otherPtr = dyn_cast<StencilFunCallExpr<DataTraits>>(other);
  return otherPtr && Expr<DataTraits>::equals(other) && callee_ == otherPtr->callee_ &&
         arguments_.size() == otherPtr->arguments_.size() &&
         std::equal(arguments_.begin(), arguments_.end(), otherPtr->arguments_.begin(),
                    [](const std::shared_ptr<Expr<DataTraits>>& a,
                       const std::shared_ptr<Expr<DataTraits>>& b) { return a->equals(b.get()); });
}

//===------------------------------------------------------------------------------------------===//
//     StencilFunArgExpr
//===------------------------------------------------------------------------------------------===//

template <typename DataTraits>
StencilFunArgExpr<DataTraits>::StencilFunArgExpr(int direction, int offset, int argumentIndex,
                                                 SourceLocation loc)
    : Expr<DataTraits>(EK_StencilFunArgExpr, loc), dimension_(direction), offset_(offset),
      argumentIndex_(argumentIndex) {}

template <typename DataTraits>
StencilFunArgExpr<DataTraits>::StencilFunArgExpr(const StencilFunArgExpr<DataTraits>& expr)
    : Expr<DataTraits>(EK_StencilFunArgExpr, expr.getSourceLocation()),
      dimension_(expr.getDimension()), offset_(expr.getOffset()),
      argumentIndex_(expr.getArgumentIndex()) {}

template <typename DataTraits>
StencilFunArgExpr<DataTraits>& StencilFunArgExpr<DataTraits>::
operator=(StencilFunArgExpr<DataTraits> expr) {
  assign(expr);
  dimension_ = expr.getDimension();
  offset_ = expr.getOffset();
  argumentIndex_ = expr.getArgumentIndex();
  return *this;
}

template <typename DataTraits>
StencilFunArgExpr<DataTraits>::~StencilFunArgExpr() {}

template <typename DataTraits>
std::shared_ptr<Expr<DataTraits>> StencilFunArgExpr<DataTraits>::clone() const {
  return std::make_shared<StencilFunArgExpr<DataTraits>>(*this);
}

template <typename DataTraits>
bool StencilFunArgExpr<DataTraits>::equals(const Expr<DataTraits>* other) const {
  const StencilFunArgExpr<DataTraits>* otherPtr = dyn_cast<StencilFunArgExpr<DataTraits>>(other);
  return otherPtr && Expr<DataTraits>::equals(other) && dimension_ == otherPtr->dimension_ &&
         offset_ == otherPtr->offset_ && argumentIndex_ == otherPtr->argumentIndex_;
}

//===------------------------------------------------------------------------------------------===//
//     VarAccessExpr
//===------------------------------------------------------------------------------------------===//

template <typename DataTraits>
VarAccessExpr<DataTraits>::VarAccessExpr(const std::string& name,
                                         std::shared_ptr<Expr<DataTraits>> index,
                                         SourceLocation loc)
    : Expr<DataTraits>(EK_VarAccessExpr, loc), name_(name), index_(index), isExternal_(false) {}

template <typename DataTraits>
VarAccessExpr<DataTraits>::VarAccessExpr(const VarAccessExpr<DataTraits>& expr)
    : Expr<DataTraits>(EK_VarAccessExpr, expr.getSourceLocation()), name_(expr.getName()),
      index_(expr.getIndex()), isExternal_(expr.isExternal()) {}

template <typename DataTraits>
VarAccessExpr<DataTraits>& VarAccessExpr<DataTraits>::operator=(VarAccessExpr<DataTraits> expr) {
  assign(expr);
  name_ = std::move(expr.getName());
  index_ = std::move(expr.getIndex());
  isExternal_ = expr.isExternal();
  return *this;
}

template <typename DataTraits>
VarAccessExpr<DataTraits>::~VarAccessExpr() {}

template <typename DataTraits>
std::shared_ptr<Expr<DataTraits>> VarAccessExpr<DataTraits>::clone() const {
  return std::make_shared<VarAccessExpr<DataTraits>>(*this);
}

template <typename DataTraits>
bool VarAccessExpr<DataTraits>::equals(const Expr<DataTraits>* other) const {
  const VarAccessExpr<DataTraits>* otherPtr = dyn_cast<VarAccessExpr<DataTraits>>(other);
  return otherPtr && Expr<DataTraits>::equals(other) && name_ == otherPtr->name_ &&
         isExternal_ == otherPtr->isExternal_ && isArrayAccess() == otherPtr->isArrayAccess() &&
         (isArrayAccess() ? index_->equals(otherPtr->index_.get()) : true);
}

template <typename DataTraits>
void VarAccessExpr<DataTraits>::replaceChildren(const std::shared_ptr<Expr<DataTraits>>& oldExpr,
                                                const std::shared_ptr<Expr<DataTraits>>& newExpr) {
  if(isArrayAccess()) {
    DAWN_ASSERT(index_ == oldExpr);
    index_ = newExpr;
  } else {
    DAWN_ASSERT_MSG((false), ("Non array vars have no children"));
  }
}

//===------------------------------------------------------------------------------------------===//
//     FieldAccessExpr
//===------------------------------------------------------------------------------------------===//

template <typename DataTraits>
FieldAccessExpr<DataTraits>::FieldAccessExpr(const std::string& name, Array3i offset,
                                             Array3i argumentMap, Array3i argumentOffset,
                                             bool negateOffset, SourceLocation loc)
    : Expr<DataTraits>(EK_FieldAccessExpr, loc), name_(name), offset_(std::move(offset)),
      argumentMap_(std::move(argumentMap)), argumentOffset_(std::move(argumentOffset)),
      negateOffset_(negateOffset) {}

template <typename DataTraits>
FieldAccessExpr<DataTraits>::FieldAccessExpr(const FieldAccessExpr<DataTraits>& expr)
    : Expr<DataTraits>(EK_FieldAccessExpr, expr.getSourceLocation()), name_(expr.getName()),
      offset_(expr.getOffset()), argumentMap_(expr.getArgumentMap()),
      argumentOffset_(expr.getArgumentOffset()), negateOffset_(expr.negateOffset()) {}

template <typename DataTraits>
FieldAccessExpr<DataTraits>& FieldAccessExpr<DataTraits>::
operator=(FieldAccessExpr<DataTraits> expr) {
  assign(expr);
  name_ = std::move(expr.getName());
  offset_ = std::move(expr.getOffset());
  argumentMap_ = std::move(expr.getArgumentMap());
  argumentOffset_ = std::move(expr.getArgumentOffset());
  negateOffset_ = expr.negateOffset();
  return *this;
}

template <typename DataTraits>
FieldAccessExpr<DataTraits>::~FieldAccessExpr() {}

template <typename DataTraits>
void FieldAccessExpr<DataTraits>::setPureOffset(const Array3i& offset) {
  offset_ = offset;
  argumentMap_ = Array3i{{-1, -1, -1}};
  argumentOffset_ = Array3i{{0, 0, 0}};
}

template <typename DataTraits>
std::shared_ptr<Expr<DataTraits>> FieldAccessExpr<DataTraits>::clone() const {
  return std::make_shared<FieldAccessExpr<DataTraits>>(*this);
}

template <typename DataTraits>
bool FieldAccessExpr<DataTraits>::equals(const Expr<DataTraits>* other) const {
  const FieldAccessExpr<DataTraits>* otherPtr = dyn_cast<FieldAccessExpr<DataTraits>>(other);
  return otherPtr && Expr<DataTraits>::equals(other) && name_ == otherPtr->name_ &&
         offset_ == otherPtr->offset_ && argumentMap_ == otherPtr->argumentMap_ &&
         argumentOffset_ == otherPtr->argumentOffset_ && negateOffset_ == otherPtr->negateOffset_;
}

//===------------------------------------------------------------------------------------------===//
//     LiteralAccessExpr
//===------------------------------------------------------------------------------------------===//

template <typename DataTraits>
LiteralAccessExpr<DataTraits>::LiteralAccessExpr(const std::string& value,
                                                 BuiltinTypeID builtinType, SourceLocation loc)
    : Expr<DataTraits>(EK_LiteralAccessExpr, loc), value_(value), builtinType_(builtinType) {}

template <typename DataTraits>
LiteralAccessExpr<DataTraits>::LiteralAccessExpr(const LiteralAccessExpr<DataTraits>& expr)
    : Expr<DataTraits>(EK_LiteralAccessExpr, expr.getSourceLocation()), value_(expr.getValue()),
      builtinType_(expr.getBuiltinType()) {}

template <typename DataTraits>
LiteralAccessExpr<DataTraits>& LiteralAccessExpr<DataTraits>::
operator=(LiteralAccessExpr<DataTraits> expr) {
  assign(expr);
  value_ = std::move(expr.getValue());
  builtinType_ = expr.getBuiltinType();
  return *this;
}

template <typename DataTraits>
LiteralAccessExpr<DataTraits>::~LiteralAccessExpr() {}

template <typename DataTraits>
std::shared_ptr<Expr<DataTraits>> LiteralAccessExpr<DataTraits>::clone() const {
  return std::make_shared<LiteralAccessExpr<DataTraits>>(*this);
}

template <typename DataTraits>
bool LiteralAccessExpr<DataTraits>::equals(const Expr<DataTraits>* other) const {
  const LiteralAccessExpr<DataTraits>* otherPtr = dyn_cast<LiteralAccessExpr<DataTraits>>(other);
  return otherPtr && Expr<DataTraits>::equals(other) && value_ == otherPtr->value_ &&
         builtinType_ == otherPtr->builtinType_;
}
} // namespace ast
} // namespace dawn
