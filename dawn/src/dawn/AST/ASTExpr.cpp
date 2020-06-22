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

#include "dawn/AST/ASTExpr.h"
#include "dawn/AST/ASTUtil.h"
#include "dawn/AST/ASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Casting.h"
#include <memory>

namespace dawn {
namespace ast {
//===------------------------------------------------------------------------------------------===//
//     UnaryOperator
//===------------------------------------------------------------------------------------------===//

UnaryOperator::UnaryOperator(const std::shared_ptr<Expr>& operand, std::string op,
                             SourceLocation loc)
    : Expr(Kind::UnaryOperator, loc), operand_(operand), op_(std::move(op)) {}

UnaryOperator::UnaryOperator(const UnaryOperator& expr)
    : Expr(Kind::UnaryOperator, expr.getSourceLocation()), operand_(expr.getOperand()->clone()),
      op_(expr.getOp()) {}

UnaryOperator& UnaryOperator::operator=(UnaryOperator expr) {
  assign(expr);
  operand_ = expr.getOperand();
  op_ = expr.getOp();
  return *this;
}

UnaryOperator::~UnaryOperator() {}

std::shared_ptr<Expr> UnaryOperator::clone() const {
  return std::make_shared<UnaryOperator>(*this);
}

bool UnaryOperator::equals(const Expr* other, bool compareData) const {
  const UnaryOperator* otherPtr = dyn_cast<UnaryOperator>(other);
  return otherPtr && Expr::equals(other, compareData) &&
         operand_->equals(otherPtr->operand_.get(), compareData) && op_ == otherPtr->op_;
}

void UnaryOperator::replaceChildren(const std::shared_ptr<Expr>& oldExpr,
                                    const std::shared_ptr<Expr>& newExpr) {
  DAWN_ASSERT(oldExpr == operand_);
  operand_ = newExpr;
}

//===------------------------------------------------------------------------------------------===//
//     BinaryOperator
//===------------------------------------------------------------------------------------------===//

BinaryOperator::BinaryOperator(const std::shared_ptr<Expr>& left, std::string op,
                               const std::shared_ptr<Expr>& right, SourceLocation loc)
    : Expr(Kind::BinaryOperator, loc), operands_{left, right}, op_(std::move(op)) {}

BinaryOperator::BinaryOperator(const BinaryOperator& expr)
    : Expr(Kind::BinaryOperator, expr.getSourceLocation()), operands_{expr.getLeft()->clone(),
                                                                      expr.getRight()->clone()},
      op_(expr.getOp()) {}

BinaryOperator& BinaryOperator::operator=(BinaryOperator expr) {
  assign(expr);
  operands_[Left] = expr.getLeft();
  operands_[Right] = expr.getRight();
  op_ = expr.getOp();
  return *this;
}

BinaryOperator::~BinaryOperator() {}

std::shared_ptr<Expr> BinaryOperator::clone() const {
  return std::make_shared<BinaryOperator>(*this);
}

bool BinaryOperator::equals(const Expr* other, bool compareData) const {
  const BinaryOperator* otherPtr = dyn_cast<BinaryOperator>(other);
  return otherPtr && Expr::equals(other, compareData) &&
         operands_[Left]->equals(otherPtr->operands_[Left].get(), compareData) &&
         operands_[Right]->equals(otherPtr->operands_[Right].get(), compareData) &&
         op_ == otherPtr->op_;
}

void BinaryOperator::replaceChildren(const std::shared_ptr<Expr>& oldExpr,
                                     const std::shared_ptr<Expr>& newExpr) {
  [[maybe_unused]] bool success = ASTHelper::replaceOperands(oldExpr, newExpr, operands_);
  DAWN_ASSERT_MSG((success), ("Expression not found"));
}

//===------------------------------------------------------------------------------------------===//
//     AssignmentExpr
//===------------------------------------------------------------------------------------------===//

AssignmentExpr::AssignmentExpr(const std::shared_ptr<Expr>& left,
                               const std::shared_ptr<Expr>& right, std::string op,
                               SourceLocation loc)
    : BinaryOperator(left, std::move(op), right, loc) {
  kind_ = Kind::AssignmentExpr;
}

AssignmentExpr::AssignmentExpr(const AssignmentExpr& expr)
    : BinaryOperator(expr.getLeft()->clone(), expr.getOp(), expr.getRight()->clone(),
                     expr.getSourceLocation()) {
  kind_ = Kind::AssignmentExpr;
}

AssignmentExpr& AssignmentExpr::operator=(AssignmentExpr expr) {
  assign(expr);
  operands_[Left] = expr.getLeft();
  operands_[Right] = expr.getRight();
  op_ = expr.getOp();
  return *this;
}

AssignmentExpr::~AssignmentExpr() {}

std::shared_ptr<Expr> AssignmentExpr::clone() const {
  return std::make_shared<AssignmentExpr>(*this);
}

bool AssignmentExpr::equals(const Expr* other, bool compareData) const {
  const AssignmentExpr* otherPtr = dyn_cast<AssignmentExpr>(other);
  return otherPtr && Expr::equals(other, compareData) &&
         operands_[Left]->equals(otherPtr->operands_[Left].get(), compareData) &&
         operands_[Right]->equals(otherPtr->operands_[Right].get(), compareData) &&
         op_ == otherPtr->op_;
}

//===------------------------------------------------------------------------------------------===//
//     NOPExpr
//===------------------------------------------------------------------------------------------===//

NOPExpr::NOPExpr(SourceLocation loc) : Expr(Kind::NOPExpr, loc) { kind_ = Kind::NOPExpr; }

NOPExpr::NOPExpr(const NOPExpr& expr) : Expr(Kind::NOPExpr, expr.getSourceLocation()) {
  kind_ = Kind::NOPExpr;
}

NOPExpr& NOPExpr::operator=(NOPExpr expr) {
  assign(expr);
  return *this;
}

NOPExpr::~NOPExpr() {}

std::shared_ptr<Expr> NOPExpr::clone() const { return std::make_shared<NOPExpr>(*this); }

bool NOPExpr::equals(const Expr* other, bool compareData) const { return true; }

//===------------------------------------------------------------------------------------------===//
//     TernaryOperator
//===------------------------------------------------------------------------------------------===//

TernaryOperator::TernaryOperator(const std::shared_ptr<Expr>& cond,
                                 const std::shared_ptr<Expr>& left,
                                 const std::shared_ptr<Expr>& right, SourceLocation loc)
    : Expr(Kind::TernaryOperator, loc), operands_{cond, left, right} {}

TernaryOperator::TernaryOperator(const TernaryOperator& expr)
    : Expr(Kind::TernaryOperator, expr.getSourceLocation()), operands_{expr.getCondition()->clone(),
                                                                       expr.getLeft()->clone(),
                                                                       expr.getRight()->clone()} {}

TernaryOperator& TernaryOperator::operator=(TernaryOperator expr) {
  assign(expr);
  operands_[Cond] = expr.getCondition();
  operands_[Left] = expr.getLeft();
  operands_[Right] = expr.getRight();
  return *this;
}

TernaryOperator::~TernaryOperator() {}

std::shared_ptr<Expr> TernaryOperator::clone() const {
  return std::make_shared<TernaryOperator>(*this);
}

bool TernaryOperator::equals(const Expr* other, bool compareData) const {
  const TernaryOperator* otherPtr = dyn_cast<TernaryOperator>(other);
  return otherPtr && Expr::equals(other, compareData) &&
         operands_[Cond]->equals(otherPtr->operands_[Cond].get(), compareData) &&
         operands_[Left]->equals(otherPtr->operands_[Left].get(), compareData) &&
         operands_[Right]->equals(otherPtr->operands_[Right].get(), compareData);
}

void TernaryOperator::replaceChildren(const std::shared_ptr<Expr>& oldExpr,
                                      const std::shared_ptr<Expr>& newExpr) {
  [[maybe_unused]] bool success = ASTHelper::replaceOperands(oldExpr, newExpr, operands_);
  DAWN_ASSERT_MSG((success), ("Expression not found"));
}

//===------------------------------------------------------------------------------------------===//
//     FunCallExpr
//===------------------------------------------------------------------------------------------===//

FunCallExpr::FunCallExpr(const std::string& callee, SourceLocation loc)
    : Expr(Kind::FunCallExpr, loc), callee_(callee) {}

FunCallExpr::FunCallExpr(const FunCallExpr& expr)
    : Expr(Kind::FunCallExpr, expr.getSourceLocation()), callee_(expr.getCallee()) {
  for(auto e : expr.getArguments())
    arguments_.push_back(e->clone());
}

FunCallExpr& FunCallExpr::operator=(FunCallExpr expr) {
  assign(expr);
  callee_ = std::move(expr.getCallee());
  arguments_ = std::move(expr.getArguments());
  return *this;
}

FunCallExpr::~FunCallExpr() {}

std::shared_ptr<Expr> FunCallExpr::clone() const { return std::make_shared<FunCallExpr>(*this); }

bool FunCallExpr::equals(const Expr* other, bool compareData) const {
  const FunCallExpr* otherPtr = dyn_cast<FunCallExpr>(other);
  return otherPtr && Expr::equals(other, compareData) && callee_ == otherPtr->callee_ &&
         arguments_.size() == otherPtr->arguments_.size() &&
         std::equal(arguments_.begin(), arguments_.end(), otherPtr->arguments_.begin(),
                    [compareData](const std::shared_ptr<Expr>& a, const std::shared_ptr<Expr>& b) {
                      return a->equals(b.get(), compareData);
                    });
}

void FunCallExpr::insertArgument(const std::shared_ptr<Expr>& expr) { arguments_.push_back(expr); }

void FunCallExpr::replaceChildren(const std::shared_ptr<Expr>& oldExpr,
                                  const std::shared_ptr<Expr>& newExpr) {
  [[maybe_unused]] bool success = ASTHelper::replaceOperands(oldExpr, newExpr, arguments_);
  DAWN_ASSERT_MSG((success), ("Expression not found"));
}

//===------------------------------------------------------------------------------------------===//
//     StencilFunCallExpr
//===------------------------------------------------------------------------------------------===//

StencilFunCallExpr::StencilFunCallExpr(const std::string& callee, SourceLocation loc)
    : FunCallExpr(callee, loc) {
  kind_ = Kind::StencilFunCallExpr;
}

StencilFunCallExpr::StencilFunCallExpr(const StencilFunCallExpr& expr)
    : FunCallExpr(expr.getCallee(), expr.getSourceLocation()) {
  kind_ = Kind::StencilFunCallExpr;
  for(auto e : expr.getArguments())
    arguments_.push_back(e->clone());
}

StencilFunCallExpr& StencilFunCallExpr::operator=(StencilFunCallExpr expr) {
  assign(expr);
  callee_ = std::move(expr.getCallee());
  arguments_ = std::move(expr.getArguments());
  return *this;
}

StencilFunCallExpr::~StencilFunCallExpr() {}

std::shared_ptr<Expr> StencilFunCallExpr::clone() const {
  return std::make_shared<StencilFunCallExpr>(*this);
}

bool StencilFunCallExpr::equals(const Expr* other, bool compareData) const {
  const StencilFunCallExpr* otherPtr = dyn_cast<StencilFunCallExpr>(other);
  return otherPtr && Expr::equals(other, compareData) && callee_ == otherPtr->callee_ &&
         arguments_.size() == otherPtr->arguments_.size() &&
         std::equal(arguments_.begin(), arguments_.end(), otherPtr->arguments_.begin(),
                    [compareData](const std::shared_ptr<Expr>& a, const std::shared_ptr<Expr>& b) {
                      return a->equals(b.get(), compareData);
                    });
}

//===------------------------------------------------------------------------------------------===//
//     StencilFunArgExpr
//===------------------------------------------------------------------------------------------===//

StencilFunArgExpr::StencilFunArgExpr(int direction, int offset, int argumentIndex,
                                     SourceLocation loc)
    : Expr(Kind::StencilFunArgExpr, loc), dimension_(direction), offset_(offset),
      argumentIndex_(argumentIndex) {}

StencilFunArgExpr::StencilFunArgExpr(const StencilFunArgExpr& expr)
    : Expr(Kind::StencilFunArgExpr, expr.getSourceLocation()), dimension_(expr.getDimension()),
      offset_(expr.getOffset()), argumentIndex_(expr.getArgumentIndex()) {}

StencilFunArgExpr& StencilFunArgExpr::operator=(StencilFunArgExpr expr) {
  assign(expr);
  dimension_ = expr.getDimension();
  offset_ = expr.getOffset();
  argumentIndex_ = expr.getArgumentIndex();
  return *this;
}

StencilFunArgExpr::~StencilFunArgExpr() {}

std::shared_ptr<Expr> StencilFunArgExpr::clone() const {
  return std::make_shared<StencilFunArgExpr>(*this);
}

bool StencilFunArgExpr::equals(const Expr* other, bool compareData) const {
  const StencilFunArgExpr* otherPtr = dyn_cast<StencilFunArgExpr>(other);
  return otherPtr && Expr::equals(other, compareData) && dimension_ == otherPtr->dimension_ &&
         offset_ == otherPtr->offset_ && argumentIndex_ == otherPtr->argumentIndex_;
}

//===------------------------------------------------------------------------------------------===//
//     VarAccessExpr
//===------------------------------------------------------------------------------------------===//

VarAccessExpr::VarAccessExpr(const std::string& name, std::shared_ptr<Expr> index,
                             SourceLocation loc)
    : Expr(Kind::VarAccessExpr, loc), name_(name), index_(index), isExternal_(false) {}

VarAccessExpr::VarAccessExpr(const VarAccessExpr& expr)
    : Expr(Kind::VarAccessExpr, expr.getSourceLocation()), name_(expr.getName()),
      index_(expr.getIndex()), isExternal_(expr.isExternal()) {
  data_ = expr.data_ ? expr.data_->clone() : nullptr;
}

VarAccessExpr& VarAccessExpr::operator=(VarAccessExpr expr) {
  assign(expr);
  data_ = expr.data_ ? expr.data_->clone() : nullptr;
  name_ = std::move(expr.getName());
  index_ = std::move(expr.getIndex());
  isExternal_ = expr.isExternal();
  return *this;
}

VarAccessExpr::~VarAccessExpr() {}

std::shared_ptr<Expr> VarAccessExpr::clone() const {
  return std::make_shared<VarAccessExpr>(*this);
}

bool VarAccessExpr::equals(const Expr* other, bool compareData) const {
  const VarAccessExpr* otherPtr = dyn_cast<VarAccessExpr>(other);
  return otherPtr && Expr::equals(other, compareData) && name_ == otherPtr->name_ &&
         isExternal_ == otherPtr->isExternal_ && isArrayAccess() == otherPtr->isArrayAccess() &&
         (isArrayAccess() ? index_->equals(otherPtr->index_.get(), compareData) : true) &&
         (compareData ? (data_ ? data_->equals(otherPtr->data_.get()) : !otherPtr->data_) : true);
}

void VarAccessExpr::replaceChildren(const std::shared_ptr<Expr>& oldExpr,
                                    const std::shared_ptr<Expr>& newExpr) {
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

FieldAccessExpr::FieldAccessExpr(const std::string& name, const Offsets& offset,
                                 Array3i argumentMap, Array3i argumentOffset, bool negateOffset,
                                 SourceLocation loc)
    : Expr(Kind::FieldAccessExpr, loc), name_(name), offset_(std::move(offset)),
      argumentMap_(std::move(argumentMap)), argumentOffset_(std::move(argumentOffset)),
      negateOffset_(negateOffset) {}

FieldAccessExpr::FieldAccessExpr(const FieldAccessExpr& expr)
    : Expr(Kind::FieldAccessExpr, expr.getSourceLocation()), name_(expr.getName()),
      offset_(expr.getOffset()), argumentMap_(expr.getArgumentMap()),
      argumentOffset_(expr.getArgumentOffset()), negateOffset_(expr.negateOffset()) {
  data_ = expr.data_ ? expr.data_->clone() : nullptr;
}

FieldAccessExpr& FieldAccessExpr::operator=(FieldAccessExpr expr) {
  assign(expr);
  data_ = expr.data_ ? expr.data_->clone() : nullptr;
  name_ = std::move(expr.getName());
  offset_ = std::move(expr.getOffset());
  argumentMap_ = std::move(expr.getArgumentMap());
  argumentOffset_ = std::move(expr.getArgumentOffset());
  negateOffset_ = expr.negateOffset();
  return *this;
}

FieldAccessExpr::~FieldAccessExpr() {}

void FieldAccessExpr::setPureOffset(const Offsets& offset) {
  offset_ = offset;
  argumentMap_ = Array3i{{-1, -1, -1}};
  argumentOffset_ = Array3i{{0, 0, 0}};
}

std::shared_ptr<Expr> FieldAccessExpr::clone() const {
  return std::make_shared<FieldAccessExpr>(*this);
}

bool FieldAccessExpr::equals(const Expr* other, bool compareData) const {
  const FieldAccessExpr* otherPtr = dyn_cast<FieldAccessExpr>(other);
  return otherPtr && Expr::equals(other, compareData) && name_ == otherPtr->name_ &&
         offset_ == otherPtr->offset_ && argumentMap_ == otherPtr->argumentMap_ &&
         argumentOffset_ == otherPtr->argumentOffset_ && negateOffset_ == otherPtr->negateOffset_ &&
         (compareData ? (data_ ? data_->equals(otherPtr->data_.get()) : !otherPtr->data_) : true);
}

//===------------------------------------------------------------------------------------------===//
//     LiteralAccessExpr
//===------------------------------------------------------------------------------------------===//

LiteralAccessExpr::LiteralAccessExpr(const std::string& value, BuiltinTypeID builtinType,
                                     SourceLocation loc)
    : Expr(Kind::LiteralAccessExpr, loc), value_(value), builtinType_(builtinType) {}

LiteralAccessExpr::LiteralAccessExpr(const LiteralAccessExpr& expr)
    : Expr(Kind::LiteralAccessExpr, expr.getSourceLocation()), value_(expr.getValue()),
      builtinType_(expr.getBuiltinType()) {
  data_ = expr.data_ ? expr.data_->clone() : nullptr;
}

LiteralAccessExpr& LiteralAccessExpr::operator=(LiteralAccessExpr expr) {
  assign(expr);
  data_ = expr.data_ ? expr.data_->clone() : nullptr;
  value_ = std::move(expr.getValue());
  builtinType_ = expr.getBuiltinType();
  return *this;
}

LiteralAccessExpr::~LiteralAccessExpr() {}

std::shared_ptr<Expr> LiteralAccessExpr::clone() const {
  return std::make_shared<LiteralAccessExpr>(*this);
}

bool LiteralAccessExpr::equals(const Expr* other, bool compareData) const {
  const LiteralAccessExpr* otherPtr = dyn_cast<LiteralAccessExpr>(other);
  return otherPtr && Expr::equals(other, compareData) && value_ == otherPtr->value_ &&
         builtinType_ == otherPtr->builtinType_ &&
         (compareData ? (data_ ? data_->equals(otherPtr->data_.get()) : !otherPtr->data_) : true);
}

//===------------------------------------------------------------------------------------------===//
//     ReductionOverNeighborExpr
//===------------------------------------------------------------------------------------------===//

bool ReductionOverNeighborExpr::chainIsValid() const {
  for(int chainIdx = 0; chainIdx < chain_.size() - 1; chainIdx++) {
    if(chain_[chainIdx] == chain_[chainIdx + 1]) {
      return false;
    }
  }
  return true;
}

ReductionOverNeighborExpr::ReductionOverNeighborExpr(std::string const& op,
                                                     std::shared_ptr<Expr> const& rhs,
                                                     std::shared_ptr<Expr> const& init,
                                                     std::vector<ast::LocationType> chain,
                                                     SourceLocation loc)
    : Expr(Kind::ReductionOverNeighborExpr, loc), op_(op), chain_(chain), operands_{rhs, init} {
  DAWN_ASSERT_MSG(chainIsValid(), "invalid neighbor chain (repeated element in succession, use "
                                  "expaneded notation (e.g. C->C becomes C->E->C\n");
}

ReductionOverNeighborExpr::ReductionOverNeighborExpr(std::string const& op,
                                                     std::shared_ptr<Expr> const& rhs,
                                                     std::shared_ptr<Expr> const& init,
                                                     std::vector<std::shared_ptr<Expr>> weights,
                                                     std::vector<ast::LocationType> chain,
                                                     SourceLocation loc)
    : Expr(Kind::ReductionOverNeighborExpr, loc), op_(op), weights_(weights),
      chain_(chain), operands_{rhs, init} {
  DAWN_ASSERT_MSG(weights.size() > 0, "empty weights vector passed!\n");
  DAWN_ASSERT_MSG(chainIsValid(), "invalid neighbor chain (repeated element in succession, use "
                                  "expaneded notation (e.g. C->C becomes C->E->C\n");
  operands_.insert(operands_.end(), weights.begin(), weights.end());
}

ReductionOverNeighborExpr::ReductionOverNeighborExpr(ReductionOverNeighborExpr const& expr)
    : Expr(Kind::ReductionOverNeighborExpr, expr.getSourceLocation()), op_(expr.getOp()),
      weights_(expr.getWeights()), chain_(expr.getNbhChain()), operands_(expr.operands_) {}

ReductionOverNeighborExpr&
ReductionOverNeighborExpr::operator=(ReductionOverNeighborExpr const& expr) {
  assign(expr);
  op_ = expr.op_;
  operands_ = expr.operands_;
  chain_ = expr.getNbhChain();
  weights_ = expr.getWeights();
  return *this;
}

std::shared_ptr<Expr> ReductionOverNeighborExpr::clone() const {
  return std::make_shared<ReductionOverNeighborExpr>(*this);
}

ArrayRef<std::shared_ptr<Expr>> ReductionOverNeighborExpr::getChildren() {
  return ExprRangeType(operands_);
} // namespace ast

bool ReductionOverNeighborExpr::equals(const Expr* other, bool compareData) const {
  const ReductionOverNeighborExpr* otherPtr = dyn_cast<ReductionOverNeighborExpr>(other);

  if(weights_.has_value()) {
    if(!otherPtr->getWeights().has_value()) {
      return false;
    }
    if(otherPtr->getWeights()->size() != weights_->size()) {
      return false;
    }
    for(int i = 0; i < weights_->size(); i++) {
      if(*weights_.value().at(i) != *otherPtr->getWeights().value().at(i)) {
        return false;
      }
    }
  }

  return otherPtr && otherPtr->getInit()->equals(getInit().get(), compareData) &&
         otherPtr->getOp() == getOp() && otherPtr->getRhs()->equals(getRhs().get(), compareData) &&
         otherPtr->getNbhChain() == getNbhChain();
}

} // namespace ast
} // namespace dawn
