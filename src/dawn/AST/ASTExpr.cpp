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
#include "dawn/Support/Assert.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/StringRef.h"
#include <memory>

namespace dawn {
namespace ast {
//===------------------------------------------------------------------------------------------===//
//     UnaryOperator
//===------------------------------------------------------------------------------------------===//

UnaryOperator::UnaryOperator(const std::shared_ptr<Expr>& operand, std::string op,
                             SourceLocation loc)
    : Expr(EK_UnaryOperator, loc), operand_(operand), op_(std::move(op)) {}

UnaryOperator::UnaryOperator(const UnaryOperator& expr)
    : Expr(EK_UnaryOperator, expr.getSourceLocation()), operand_(expr.getOperand()->clone()),
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

bool UnaryOperator::equals(const Expr* other) const {
  const UnaryOperator* otherPtr = dyn_cast<UnaryOperator>(other);
  return otherPtr && Expr::equals(other) && operand_->equals(otherPtr->operand_.get()) &&
         op_ == otherPtr->op_;
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
    : Expr(EK_BinaryOperator, loc), operands_{left, right}, op_(std::move(op)) {}

BinaryOperator::BinaryOperator(const BinaryOperator& expr)
    : Expr(EK_BinaryOperator, expr.getSourceLocation()), operands_{expr.getLeft()->clone(),
                                                                   expr.getRight()->clone()},
      op_(expr.getOp()) {}

BinaryOperator& BinaryOperator::operator=(BinaryOperator expr) {
  assign(expr);
  operands_[OK_Left] = expr.getLeft();
  operands_[OK_Right] = expr.getRight();
  op_ = expr.getOp();
  return *this;
}

BinaryOperator::~BinaryOperator() {}

std::shared_ptr<Expr> BinaryOperator::clone() const {
  return std::make_shared<BinaryOperator>(*this);
}

bool BinaryOperator::equals(const Expr* other) const {
  const BinaryOperator* otherPtr = dyn_cast<BinaryOperator>(other);
  return otherPtr && Expr::equals(other) &&
         operands_[OK_Left]->equals(otherPtr->operands_[OK_Left].get()) &&
         operands_[OK_Right]->equals(otherPtr->operands_[OK_Right].get()) && op_ == otherPtr->op_;
}

void BinaryOperator::replaceChildren(const std::shared_ptr<Expr>& oldExpr,
                                     const std::shared_ptr<Expr>& newExpr) {
  bool success = ASTHelper::replaceOperands(oldExpr, newExpr, operands_);
  DAWN_ASSERT_MSG((success), ("Expression not found"));
}

//===------------------------------------------------------------------------------------------===//
//     AssignmentExpr
//===------------------------------------------------------------------------------------------===//

AssignmentExpr::AssignmentExpr(const std::shared_ptr<Expr>& left,
                               const std::shared_ptr<Expr>& right, std::string op,
                               SourceLocation loc)
    : BinaryOperator(left, std::move(op), right, loc) {
  kind_ = EK_AssignmentExpr;
}

AssignmentExpr::AssignmentExpr(const AssignmentExpr& expr)
    : BinaryOperator(expr.getLeft()->clone(), expr.getOp(), expr.getRight()->clone(),
                     expr.getSourceLocation()) {
  kind_ = EK_AssignmentExpr;
}

AssignmentExpr& AssignmentExpr::operator=(AssignmentExpr expr) {
  assign(expr);
  operands_[OK_Left] = expr.getLeft();
  operands_[OK_Right] = expr.getRight();
  op_ = expr.getOp();
  return *this;
}

AssignmentExpr::~AssignmentExpr() {}

std::shared_ptr<Expr> AssignmentExpr::clone() const {
  return std::make_shared<AssignmentExpr>(*this);
}

bool AssignmentExpr::equals(const Expr* other) const {
  const AssignmentExpr* otherPtr = dyn_cast<AssignmentExpr>(other);
  return otherPtr && Expr::equals(other) &&
         operands_[OK_Left]->equals(otherPtr->operands_[OK_Left].get()) &&
         operands_[OK_Right]->equals(otherPtr->operands_[OK_Right].get()) && op_ == otherPtr->op_;
}

//===------------------------------------------------------------------------------------------===//
//     NOPExpr
//===------------------------------------------------------------------------------------------===//

NOPExpr::NOPExpr(SourceLocation loc) : Expr(EK_NOPExpr, loc) { kind_ = EK_NOPExpr; }

NOPExpr::NOPExpr(const NOPExpr& expr) : Expr(EK_NOPExpr, expr.getSourceLocation()) {
  kind_ = EK_NOPExpr;
}

NOPExpr& NOPExpr::operator=(NOPExpr expr) {
  assign(expr);
  return *this;
}

NOPExpr::~NOPExpr() {}

std::shared_ptr<Expr> NOPExpr::clone() const { return std::make_shared<NOPExpr>(*this); }

bool NOPExpr::equals(const Expr* other) const { return true; }

//===------------------------------------------------------------------------------------------===//
//     TernaryOperator
//===------------------------------------------------------------------------------------------===//

TernaryOperator::TernaryOperator(const std::shared_ptr<Expr>& cond,
                                 const std::shared_ptr<Expr>& left,
                                 const std::shared_ptr<Expr>& right, SourceLocation loc)
    : Expr(EK_TernaryOperator, loc), operands_{cond, left, right} {}

TernaryOperator::TernaryOperator(const TernaryOperator& expr)
    : Expr(EK_TernaryOperator, expr.getSourceLocation()), operands_{expr.getCondition()->clone(),
                                                                    expr.getLeft()->clone(),
                                                                    expr.getRight()->clone()} {}

TernaryOperator& TernaryOperator::operator=(TernaryOperator expr) {
  assign(expr);
  operands_[OK_Cond] = expr.getCondition();
  operands_[OK_Left] = expr.getLeft();
  operands_[OK_Right] = expr.getRight();
  return *this;
}

TernaryOperator::~TernaryOperator() {}

std::shared_ptr<Expr> TernaryOperator::clone() const {
  return std::make_shared<TernaryOperator>(*this);
}

bool TernaryOperator::equals(const Expr* other) const {
  const TernaryOperator* otherPtr = dyn_cast<TernaryOperator>(other);
  return otherPtr && Expr::equals(other) &&
         operands_[OK_Cond]->equals(otherPtr->operands_[OK_Cond].get()) &&
         operands_[OK_Left]->equals(otherPtr->operands_[OK_Left].get()) &&
         operands_[OK_Right]->equals(otherPtr->operands_[OK_Right].get());
}

void TernaryOperator::replaceChildren(const std::shared_ptr<Expr>& oldExpr,
                                      const std::shared_ptr<Expr>& newExpr) {
  bool success = ASTHelper::replaceOperands(oldExpr, newExpr, operands_);
  DAWN_ASSERT_MSG((success), ("Expression not found"));
}

//===------------------------------------------------------------------------------------------===//
//     FunCallExpr
//===------------------------------------------------------------------------------------------===//

FunCallExpr::FunCallExpr(const std::string& callee, SourceLocation loc)
    : Expr(EK_FunCallExpr, loc), callee_(callee) {}

FunCallExpr::FunCallExpr(const FunCallExpr& expr)
    : Expr(EK_FunCallExpr, expr.getSourceLocation()), callee_(expr.getCallee()) {
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

bool FunCallExpr::equals(const Expr* other) const {
  const FunCallExpr* otherPtr = dyn_cast<FunCallExpr>(other);
  return otherPtr && Expr::equals(other) && callee_ == otherPtr->callee_ &&
         arguments_.size() == otherPtr->arguments_.size() &&
         std::equal(arguments_.begin(), arguments_.end(), otherPtr->arguments_.begin(),
                    [](const std::shared_ptr<Expr>& a, const std::shared_ptr<Expr>& b) {
                      return a->equals(b.get());
                    });
}

void FunCallExpr::insertArgument(const std::shared_ptr<Expr>& expr) { arguments_.push_back(expr); }

void FunCallExpr::replaceChildren(const std::shared_ptr<Expr>& oldExpr,
                                  const std::shared_ptr<Expr>& newExpr) {
  bool success = ASTHelper::replaceOperands(oldExpr, newExpr, arguments_);
  DAWN_ASSERT_MSG((success), ("Expression not found"));
}

//===------------------------------------------------------------------------------------------===//
//     StencilFunCallExpr
//===------------------------------------------------------------------------------------------===//

StencilFunCallExpr::StencilFunCallExpr(const std::string& callee, SourceLocation loc)
    : FunCallExpr(callee, loc) {
  kind_ = EK_StencilFunCallExpr;
}

StencilFunCallExpr::StencilFunCallExpr(const StencilFunCallExpr& expr)
    : FunCallExpr(expr.getCallee(), expr.getSourceLocation()) {
  kind_ = EK_StencilFunCallExpr;
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

bool StencilFunCallExpr::equals(const Expr* other) const {
  const StencilFunCallExpr* otherPtr = dyn_cast<StencilFunCallExpr>(other);
  return otherPtr && Expr::equals(other) && callee_ == otherPtr->callee_ &&
         arguments_.size() == otherPtr->arguments_.size() &&
         std::equal(arguments_.begin(), arguments_.end(), otherPtr->arguments_.begin(),
                    [](const std::shared_ptr<Expr>& a, const std::shared_ptr<Expr>& b) {
                      return a->equals(b.get());
                    });
}

//===------------------------------------------------------------------------------------------===//
//     StencilFunArgExpr
//===------------------------------------------------------------------------------------------===//

StencilFunArgExpr::StencilFunArgExpr(int direction, int offset, int argumentIndex,
                                     SourceLocation loc)
    : Expr(EK_StencilFunArgExpr, loc), dimension_(direction), offset_(offset),
      argumentIndex_(argumentIndex) {}

StencilFunArgExpr::StencilFunArgExpr(const StencilFunArgExpr& expr)
    : Expr(EK_StencilFunArgExpr, expr.getSourceLocation()), dimension_(expr.getDimension()),
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

bool StencilFunArgExpr::equals(const Expr* other) const {
  const StencilFunArgExpr* otherPtr = dyn_cast<StencilFunArgExpr>(other);
  return otherPtr && Expr::equals(other) && dimension_ == otherPtr->dimension_ &&
         offset_ == otherPtr->offset_ && argumentIndex_ == otherPtr->argumentIndex_;
}

//===------------------------------------------------------------------------------------------===//
//     VarAccessExpr
//===------------------------------------------------------------------------------------------===//

VarAccessExpr::VarAccessExpr(const std::string& name, std::shared_ptr<Expr> index,
                             SourceLocation loc)
    : Expr(EK_VarAccessExpr, loc), name_(name), index_(index), isExternal_(false) {}

VarAccessExpr::VarAccessExpr(const VarAccessExpr& expr)
    : Expr(EK_VarAccessExpr, expr.getSourceLocation()), name_(expr.getName()),
      index_(expr.getIndex()), isExternal_(expr.isExternal()) {}

VarAccessExpr& VarAccessExpr::operator=(VarAccessExpr expr) {
  assign(expr);
  name_ = std::move(expr.getName());
  index_ = std::move(expr.getIndex());
  isExternal_ = expr.isExternal();
  return *this;
}

VarAccessExpr::~VarAccessExpr() {}

std::shared_ptr<Expr> VarAccessExpr::clone() const {
  return std::make_shared<VarAccessExpr>(*this);
}

bool VarAccessExpr::equals(const Expr* other) const {
  const VarAccessExpr* otherPtr = dyn_cast<VarAccessExpr>(other);
  return otherPtr && Expr::equals(other) && name_ == otherPtr->name_ &&
         isExternal_ == otherPtr->isExternal_ && isArrayAccess() == otherPtr->isArrayAccess() &&
         (isArrayAccess() ? index_->equals(otherPtr->index_.get()) : true);
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

FieldAccessExpr::FieldAccessExpr(const std::string& name, Array3i offset, Array3i argumentMap,
                                 Array3i argumentOffset, bool negateOffset, SourceLocation loc)
    : Expr(EK_FieldAccessExpr, loc), name_(name), offset_(std::move(offset)),
      argumentMap_(std::move(argumentMap)), argumentOffset_(std::move(argumentOffset)),
      negateOffset_(negateOffset) {}

FieldAccessExpr::FieldAccessExpr(const FieldAccessExpr& expr)
    : Expr(EK_FieldAccessExpr, expr.getSourceLocation()), name_(expr.getName()),
      offset_(expr.getOffset()), argumentMap_(expr.getArgumentMap()),
      argumentOffset_(expr.getArgumentOffset()), negateOffset_(expr.negateOffset()) {}

FieldAccessExpr& FieldAccessExpr::operator=(FieldAccessExpr expr) {
  assign(expr);
  name_ = std::move(expr.getName());
  offset_ = std::move(expr.getOffset());
  argumentMap_ = std::move(expr.getArgumentMap());
  argumentOffset_ = std::move(expr.getArgumentOffset());
  negateOffset_ = expr.negateOffset();
  return *this;
}

FieldAccessExpr::~FieldAccessExpr() {}

void FieldAccessExpr::setPureOffset(const Array3i& offset) {
  offset_ = offset;
  argumentMap_ = Array3i{{-1, -1, -1}};
  argumentOffset_ = Array3i{{0, 0, 0}};
}

std::shared_ptr<Expr> FieldAccessExpr::clone() const {
  return std::make_shared<FieldAccessExpr>(*this);
}

bool FieldAccessExpr::equals(const Expr* other) const {
  const FieldAccessExpr* otherPtr = dyn_cast<FieldAccessExpr>(other);
  return otherPtr && Expr::equals(other) && name_ == otherPtr->name_ &&
         offset_ == otherPtr->offset_ && argumentMap_ == otherPtr->argumentMap_ &&
         argumentOffset_ == otherPtr->argumentOffset_ && negateOffset_ == otherPtr->negateOffset_;
}

//===------------------------------------------------------------------------------------------===//
//     LiteralAccessExpr
//===------------------------------------------------------------------------------------------===//

LiteralAccessExpr::LiteralAccessExpr(const std::string& value, BuiltinTypeID builtinType,
                                     SourceLocation loc)
    : Expr(EK_LiteralAccessExpr, loc), value_(value), builtinType_(builtinType) {}

LiteralAccessExpr::LiteralAccessExpr(const LiteralAccessExpr& expr)
    : Expr(EK_LiteralAccessExpr, expr.getSourceLocation()), value_(expr.getValue()),
      builtinType_(expr.getBuiltinType()) {}

LiteralAccessExpr& LiteralAccessExpr::operator=(LiteralAccessExpr expr) {
  assign(expr);
  value_ = std::move(expr.getValue());
  builtinType_ = expr.getBuiltinType();
  return *this;
}

LiteralAccessExpr::~LiteralAccessExpr() {}

std::shared_ptr<Expr> LiteralAccessExpr::clone() const {
  return std::make_shared<LiteralAccessExpr>(*this);
}

bool LiteralAccessExpr::equals(const Expr* other) const {
  const LiteralAccessExpr* otherPtr = dyn_cast<LiteralAccessExpr>(other);
  return otherPtr && Expr::equals(other) && value_ == otherPtr->value_ &&
         builtinType_ == otherPtr->builtinType_;
}

ReductionOverNeighborExpr::ReductionOverNeighborExpr(std::string const& op,
                                                     std::shared_ptr<Expr> rhs,
                                                     std::shared_ptr<LiteralAccessExpr> init,
                                                     SourceLocation loc)
    : Expr(EK_ReductionOverNeighborExpr, loc), op_(op), rhs_(std::move(rhs)),
      init_(std::move(init)) {}

ReductionOverNeighborExpr::ReductionOverNeighborExpr(const ReductionOverNeighborExpr& expr)
    : Expr(EK_ReductionOverNeighborExpr, expr.getSourceLocation()), op_(expr.op_),
      rhs_(expr.rhs_->clone()),
      init_(std::dynamic_pointer_cast<LiteralAccessExpr>(expr.init_->clone())) {}

ReductionOverNeighborExpr& ReductionOverNeighborExpr::operator=(ReductionOverNeighborExpr stmt) {
  assign(stmt);
  init_ = stmt.init_;
  op_ = stmt.op_;
  rhs_ = stmt.rhs_;
  return *this;
}

std::shared_ptr<Expr> ReductionOverNeighborExpr::clone() const {
  return std::make_shared<ReductionOverNeighborExpr>(*this);
}

bool ReductionOverNeighborExpr::equals(const Expr* other) const {
  const ReductionOverNeighborExpr* otherPtr = dyn_cast<ReductionOverNeighborExpr>(other);
  return otherPtr && otherPtr->getInit() == getInit() && otherPtr->getOp() == getOp() &&
         otherPtr->getRhs() == getRhs();
}
} // namespace ast
} // namespace dawn
