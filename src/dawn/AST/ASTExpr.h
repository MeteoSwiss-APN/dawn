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

#ifndef DAWN_AST_ASTEXPR_H
#define DAWN_AST_ASTEXPR_H

#include "dawn/AST/ASTVisitorHelpers.h"
#include "dawn/Support/Array.h"
#include "dawn/Support/ArrayRef.h"
#include "dawn/Support/SourceLocation.h"
#include "dawn/Support/Type.h"
#include "dawn/Support/UIDGenerator.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dawn {
namespace ast {

/// @brief Abstract base class of all expressions
/// @ingroup ast
template <typename DataTraits>
class Expr : public std::enable_shared_from_this<Expr<DataTraits>> {
public:
  /// @brief Discriminator for RTTI (dyn_cast<> et al.)
  enum ExprKind {
    EK_UnaryOperator,
    EK_BinaryOperator,
    EK_AssignmentExpr,
    EK_TernaryOperator,
    EK_FunCallExpr,
    EK_StencilFunCallExpr,
    EK_StencilFunArgExpr,
    EK_VarAccessExpr,
    EK_FieldAccessExpr,
    EK_LiteralAccessExpr,
    EK_NOPExpr,
  };

  using ExprRangeType = ArrayRef<std::shared_ptr<Expr<DataTraits>>>;

  /// @name Constructor & Destructor
  /// @{
  Expr(ExprKind kind, SourceLocation loc = SourceLocation())
      : kind_(kind), loc_(loc), expressionID_(UIDGenerator::getInstance()->get()) {}
  virtual ~Expr() {}
  /// @}

  /// @brief Hook for Visitors
  virtual void accept(ASTVisitor<DataTraits>& visitor) = 0;
  virtual void accept(ASTVisitorNonConst<DataTraits>& visitor) = 0;
  virtual void accept(ASTVisitorForwarding<DataTraits>& visitor) = 0;
  virtual std::shared_ptr<Expr<DataTraits>>
  acceptAndReplace(ASTVisitorPostOrder<DataTraits>& visitor) = 0;
  virtual void accept(ASTVisitorForwardingNonConst<DataTraits>& visitor) = 0;
  virtual void accept(ASTVisitorDisabled<DataTraits>& visitor) = 0;

  /// @brief Clone the current expression
  virtual std::shared_ptr<Expr<DataTraits>> clone() const = 0;

  /// @brief Get kind of Expr (used by RTTI dyn_cast<> et al.)
  ExprKind getKind() const { return kind_; }

  /// @brief Get original source location
  const SourceLocation& getSourceLocation() const { return loc_; }

  /// @brief Iterate children (if any)
  virtual ExprRangeType getChildren() { return ExprRangeType(); }

  virtual void replaceChildren(const std::shared_ptr<Expr<DataTraits>>& old_,
                               const std::shared_ptr<Expr<DataTraits>>& new_) {
    DAWN_ASSERT(false);
  }

  /// @brief Compare for equality
  /// @{
  virtual bool equals(const std::shared_ptr<Expr<DataTraits>>& other) const {
    return equals(other.get());
  }
  virtual bool equals(const Expr<DataTraits>* other) const { return kind_ == other->kind_; }
  /// @}

  /// @name Operators
  /// @{
  bool operator==(const Expr<DataTraits>& other) const { return other.equals(this); }
  bool operator!=(const Expr<DataTraits>& other) const { return !(*this == other); }
  /// @}

  /// @brief get the expressionID for mapping
  int getID() const { return expressionID_; }

  void setID(int id) { expressionID_ = id; }

protected:
  void assign(const Expr<DataTraits>& other) {
    kind_ = other.kind_;
    loc_ = other.loc_;
  }

protected:
  ExprKind kind_;
  SourceLocation loc_;

  int expressionID_;
};

//  Need to explicitly specify names of base class in derived classes due to templating.
#define USING_EXPR_BASE_NAMES                                                                      \
  using typename Expr<DataTraits>::ExprRangeType;                                                  \
  using typename Expr<DataTraits>::ExprKind;                                                       \
  using Expr<DataTraits>::EK_UnaryOperator;                                                        \
  using Expr<DataTraits>::EK_BinaryOperator;                                                       \
  using Expr<DataTraits>::EK_AssignmentExpr;                                                       \
  using Expr<DataTraits>::EK_TernaryOperator;                                                      \
  using Expr<DataTraits>::EK_FunCallExpr;                                                          \
  using Expr<DataTraits>::EK_StencilFunCallExpr;                                                   \
  using Expr<DataTraits>::EK_StencilFunArgExpr;                                                    \
  using Expr<DataTraits>::EK_VarAccessExpr;                                                        \
  using Expr<DataTraits>::EK_FieldAccessExpr;                                                      \
  using Expr<DataTraits>::EK_LiteralAccessExpr;                                                    \
  using Expr<DataTraits>::EK_NOPExpr;                                                              \
  using Expr<DataTraits>::kind_;                                                                   \
  using Expr<DataTraits>::loc_;                                                                    \
  using Expr<DataTraits>::expressionID_;

//===------------------------------------------------------------------------------------------===//
//     UnaryOperator
//===------------------------------------------------------------------------------------------===//

/// @brief Unary Operations (i.e `op operand`)
/// @ingroup ast
template <typename DataTraits>
class UnaryOperator : public DataTraits::UnaryOperator, public Expr<DataTraits> {
protected:
  std::shared_ptr<Expr<DataTraits>> operand_;
  std::string op_;

public:
  USING_EXPR_BASE_NAMES

  /// @name Constructor & Destructor
  /// @{
  UnaryOperator(const std::shared_ptr<Expr<DataTraits>>& operand, std::string op,
                SourceLocation loc = SourceLocation());
  UnaryOperator(const UnaryOperator<DataTraits>& expr);
  UnaryOperator<DataTraits>& operator=(UnaryOperator<DataTraits> expr);
  virtual ~UnaryOperator();
  /// @}

  void setOperand(const std::shared_ptr<Expr<DataTraits>>& operand) { operand_ = operand; }
  const std::shared_ptr<Expr<DataTraits>>& getOperand() const { return operand_; }
  const char* getOp() const { return op_.c_str(); }

  virtual std::shared_ptr<Expr<DataTraits>> clone() const override;
  virtual bool equals(const Expr<DataTraits>* other) const override;
  static bool classof(const Expr<DataTraits>* expr) { return expr->getKind() == EK_UnaryOperator; }
  virtual ExprRangeType getChildren() override { return ExprRangeType(operand_); }
  virtual void replaceChildren(const std::shared_ptr<Expr<DataTraits>>& oldExpr,
                               const std::shared_ptr<Expr<DataTraits>>& newExpr) override;
  ACCEPTVISITOR(Expr<DataTraits>, UnaryOperator<DataTraits>)
};

//===------------------------------------------------------------------------------------------===//
//     BinaryOperator
//===------------------------------------------------------------------------------------------===//

/// @brief Binary Operations (i.e `left op right`)
/// @ingroup ast
template <typename DataTraits>
class BinaryOperator : public DataTraits::BinaryOperator, public Expr<DataTraits> {
protected:
  enum OperandKind { OK_Left = 0, OK_Right };
  std::array<std::shared_ptr<Expr<DataTraits>>, 2> operands_;
  std::string op_;

public:
  USING_EXPR_BASE_NAMES

  /// @name Constructor & Destructor
  /// @{
  BinaryOperator(const std::shared_ptr<Expr<DataTraits>>& left, std::string op,
                 const std::shared_ptr<Expr<DataTraits>>& right,
                 SourceLocation loc = SourceLocation());
  BinaryOperator(const BinaryOperator<DataTraits>& expr);
  BinaryOperator<DataTraits>& operator=(BinaryOperator<DataTraits> expr);
  virtual ~BinaryOperator();
  /// @}

  void setLeft(const std::shared_ptr<Expr<DataTraits>>& left) { operands_[OK_Left] = left; }
  const std::shared_ptr<Expr<DataTraits>>& getLeft() const { return operands_[OK_Left]; }
  std::shared_ptr<Expr<DataTraits>>& getLeft() { return operands_[OK_Left]; }

  void setRight(const std::shared_ptr<Expr<DataTraits>>& right) { operands_[OK_Right] = right; }
  const std::shared_ptr<Expr<DataTraits>>& getRight() const { return operands_[OK_Right]; }
  std::shared_ptr<Expr<DataTraits>>& getRight() { return operands_[OK_Right]; }

  const char* getOp() const { return op_.c_str(); }

  virtual std::shared_ptr<Expr<DataTraits>> clone() const override;
  virtual bool equals(const Expr<DataTraits>* other) const override;
  static bool classof(const Expr<DataTraits>* expr) { return expr->getKind() == EK_BinaryOperator; }
  virtual ExprRangeType getChildren() override { return ExprRangeType(operands_); }
  virtual void replaceChildren(const std::shared_ptr<Expr<DataTraits>>& oldExpr,
                               const std::shared_ptr<Expr<DataTraits>>& newExpr) override;

  ACCEPTVISITOR(Expr<DataTraits>, BinaryOperator<DataTraits>)
};

//===------------------------------------------------------------------------------------------===//
//     AssignmentExpr
//===------------------------------------------------------------------------------------------===//

/// @brief Assignment expression (i.e `left = right`)
/// @ingroup ast
template <typename DataTraits>
class AssignmentExpr : public DataTraits::AssignmentExpr, public BinaryOperator<DataTraits> {
public:
  USING_EXPR_BASE_NAMES
  using typename BinaryOperator<DataTraits>::OperandKind;
  using BinaryOperator<DataTraits>::OK_Left;
  using BinaryOperator<DataTraits>::OK_Right;
  using BinaryOperator<DataTraits>::operands_;
  using BinaryOperator<DataTraits>::op_;

  /// @name Constructor & Destructor
  /// @{
  AssignmentExpr(const std::shared_ptr<Expr<DataTraits>>& left,
                 const std::shared_ptr<Expr<DataTraits>>& right, std::string op = "=",
                 SourceLocation loc = SourceLocation());
  AssignmentExpr(const AssignmentExpr<DataTraits>& expr);
  AssignmentExpr<DataTraits>& operator=(AssignmentExpr<DataTraits> expr);
  virtual ~AssignmentExpr();
  /// @}

  virtual std::shared_ptr<Expr<DataTraits>> clone() const override;
  virtual bool equals(const Expr<DataTraits>* other) const override;
  static bool classof(const Expr<DataTraits>* expr) { return expr->getKind() == EK_AssignmentExpr; }
  ACCEPTVISITOR(Expr<DataTraits>, AssignmentExpr<DataTraits>)
};

//===------------------------------------------------------------------------------------------===//
//     AssignmentExpr
//===------------------------------------------------------------------------------------------===//

/// @brief NOP expression
/// @ingroup ast
template <typename DataTraits>
class NOPExpr : public DataTraits::NOPExpr, public Expr<DataTraits> {
public:
  USING_EXPR_BASE_NAMES

  /// @name Constructor & Destructor
  /// @{
  NOPExpr(SourceLocation loc = SourceLocation());
  NOPExpr(const NOPExpr<DataTraits>& expr);
  NOPExpr<DataTraits>& operator=(NOPExpr<DataTraits> expr);
  virtual ~NOPExpr();
  /// @}

  virtual std::shared_ptr<Expr<DataTraits>> clone() const override;
  virtual bool equals(const Expr<DataTraits>* other) const override;
  static bool classof(const Expr<DataTraits>* expr) { return expr->getKind() == EK_NOPExpr; }
  ACCEPTVISITOR(Expr<DataTraits>, NOPExpr<DataTraits>)
};

//===------------------------------------------------------------------------------------------===//
//     TernaryOperator
//===------------------------------------------------------------------------------------------===//

/// @brief Ternary Operations (i.e `condition ? left : right`)
/// @ingroup ast
template <typename DataTraits>
class TernaryOperator : public DataTraits::TernaryOperator, public Expr<DataTraits> {
protected:
  enum OperandKind { OK_Cond = 0, OK_Left, OK_Right };
  std::array<std::shared_ptr<Expr<DataTraits>>, 3> operands_;

public:
  USING_EXPR_BASE_NAMES

  /// @name Constructor & Destructor
  /// @{
  TernaryOperator(const std::shared_ptr<Expr<DataTraits>>& cond,
                  const std::shared_ptr<Expr<DataTraits>>& left,
                  const std::shared_ptr<Expr<DataTraits>>& right,
                  SourceLocation loc = SourceLocation());
  TernaryOperator(const TernaryOperator<DataTraits>& expr);
  TernaryOperator<DataTraits>& operator=(TernaryOperator<DataTraits> expr);
  virtual ~TernaryOperator();
  /// @}

  void setCondition(const std::shared_ptr<Expr<DataTraits>>& condition) {
    operands_[OK_Cond] = condition;
  }
  const std::shared_ptr<Expr<DataTraits>>& getCondition() const { return operands_[OK_Cond]; }
  std::shared_ptr<Expr<DataTraits>>& getCondition() { return operands_[OK_Cond]; }

  void setLeft(const std::shared_ptr<Expr<DataTraits>>& left) { operands_[OK_Left] = left; }
  const std::shared_ptr<Expr<DataTraits>>& getLeft() const { return operands_[OK_Left]; }
  std::shared_ptr<Expr<DataTraits>>& getLeft() { return operands_[OK_Left]; }

  void setRight(const std::shared_ptr<Expr<DataTraits>>& right) { operands_[OK_Right] = right; }
  const std::shared_ptr<Expr<DataTraits>>& getRight() const { return operands_[OK_Right]; }
  std::shared_ptr<Expr<DataTraits>>& getRight() { return operands_[OK_Right]; }

  const char* getOp() const { return "?"; }
  const char* getSeperator() const { return ":"; }

  virtual std::shared_ptr<Expr<DataTraits>> clone() const override;
  virtual bool equals(const Expr<DataTraits>* other) const override;
  static bool classof(const Expr<DataTraits>* expr) {
    return expr->getKind() == EK_TernaryOperator;
  }
  virtual ExprRangeType getChildren() override { return ExprRangeType(operands_); }
  virtual void replaceChildren(const std::shared_ptr<Expr<DataTraits>>& oldExpr,
                               const std::shared_ptr<Expr<DataTraits>>& newExpr) override;
  ACCEPTVISITOR(Expr<DataTraits>, TernaryOperator<DataTraits>)
};

//===------------------------------------------------------------------------------------------===//
//     FunCallExpr
//===------------------------------------------------------------------------------------------===//

/// @brief Call expression
/// @ingroup ast
template <typename DataTraits>
class FunCallExpr : public DataTraits::FunCallExpr, public Expr<DataTraits> {
public:
  USING_EXPR_BASE_NAMES
  using ArgumentsList = std::vector<std::shared_ptr<Expr<DataTraits>>>;

  /// @name Constructor & Destructor
  /// @{
  FunCallExpr(const std::string& callee, SourceLocation loc = SourceLocation());
  FunCallExpr(const FunCallExpr<DataTraits>& expr);
  FunCallExpr<DataTraits>& operator=(FunCallExpr<DataTraits> expr);
  virtual ~FunCallExpr();
  /// @}

  std::string& getCallee() { return callee_; }
  const std::string& getCallee() const { return callee_; }

  ArgumentsList& getArguments() { return arguments_; }
  const ArgumentsList& getArguments() const { return arguments_; }

  void setCallee(std::string name) { callee_ = name; }

  void insertArgument(const std::shared_ptr<Expr<DataTraits>>& expr);

  template <typename Iterator>
  inline void insertArguments(Iterator begin, Iterator end) {
    for(Iterator it = begin; it != end; ++it) {
      insertArgument(*it);
    }
  }

  virtual std::shared_ptr<Expr<DataTraits>> clone() const override;
  virtual bool equals(const Expr<DataTraits>* other) const override;
  static bool classof(const Expr<DataTraits>* expr) { return expr->getKind() == EK_FunCallExpr; }
  virtual ExprRangeType getChildren() override { return ExprRangeType(arguments_); }
  virtual void replaceChildren(const std::shared_ptr<Expr<DataTraits>>& oldExpr,
                               const std::shared_ptr<Expr<DataTraits>>& newExpr) override;
  ACCEPTVISITOR(Expr<DataTraits>, FunCallExpr<DataTraits>)
protected:
  std::string callee_;
  ArgumentsList arguments_;
};

//===------------------------------------------------------------------------------------------===//
//     StencilFunCallExpr
//===------------------------------------------------------------------------------------------===//

/// @brief Stencil function call
/// @ingroup ast
template <typename DataTraits>
class StencilFunCallExpr : public DataTraits::StencilFunCallExpr, public FunCallExpr<DataTraits> {
public:
  USING_EXPR_BASE_NAMES
  using typename FunCallExpr<DataTraits>::ArgumentsList;

  /// @name Constructor & Destructor
  /// @{
  StencilFunCallExpr(const std::string& callee, SourceLocation loc = SourceLocation());
  StencilFunCallExpr(const StencilFunCallExpr<DataTraits>& expr);
  StencilFunCallExpr<DataTraits>& operator=(StencilFunCallExpr<DataTraits> expr);
  virtual ~StencilFunCallExpr();
  /// @}

  //  void setName(std::string name);
  virtual std::shared_ptr<Expr<DataTraits>> clone() const override;
  virtual bool equals(const Expr<DataTraits>* other) const override;
  static bool classof(const Expr<DataTraits>* expr) {
    return expr->getKind() == EK_StencilFunCallExpr;
  }
  ACCEPTVISITOR(Expr<DataTraits>, StencilFunCallExpr<DataTraits>)
protected:
  using FunCallExpr<DataTraits>::callee_;
  using FunCallExpr<DataTraits>::arguments_;
};

//===------------------------------------------------------------------------------------------===//
//     StencilFunArgExpr
//===------------------------------------------------------------------------------------------===//

/// @brief Stencil function argument (direction or offset)
/// @ingroup ast
template <typename DataTraits>
class StencilFunArgExpr : public DataTraits::StencilFunArgExpr, public Expr<DataTraits> {
  int dimension_;
  int offset_;
  int argumentIndex_;

public:
  USING_EXPR_BASE_NAMES

  /// @name Constructor & Destructor
  /// @{

  /// @brief Construct a StencilFunArgExpr
  ///
  /// @param direction      Direction, i.e 0 for I, 1 for J and 2 for K) or -1 if no directional
  ///                       argument is set
  /// @param offset         Offset to the provided direction or argument
  /// @param argumentIndex  In nested stencil function call this references the argument of the
  ///                       outer stencil function
  StencilFunArgExpr(int direction, int offset, int argumentIndex,
                    SourceLocation loc = SourceLocation());
  StencilFunArgExpr(const StencilFunArgExpr<DataTraits>& expr);
  StencilFunArgExpr<DataTraits>& operator=(StencilFunArgExpr<DataTraits> expr);
  virtual ~StencilFunArgExpr();
  /// @}

  bool needsLazyEval() const { return argumentIndex_ != -1; }

  int getDimension() const { return dimension_; }
  int getOffset() const { return offset_; }
  int getArgumentIndex() const { return argumentIndex_; }

  virtual std::shared_ptr<Expr<DataTraits>> clone() const override;
  virtual bool equals(const Expr<DataTraits>* other) const override;
  static bool classof(const Expr<DataTraits>* expr) {
    return expr->getKind() == EK_StencilFunArgExpr;
  }
  ACCEPTVISITOR(Expr<DataTraits>, StencilFunArgExpr<DataTraits>)
};

//===------------------------------------------------------------------------------------------===//
//     VarAccessExpr
//===------------------------------------------------------------------------------------------===//

/// @brief Variable access expression
/// @ingroup ast
template <typename DataTraits>
class VarAccessExpr : public DataTraits::VarAccessExpr, public Expr<DataTraits> {
  std::string name_;
  std::shared_ptr<Expr<DataTraits>> index_;
  bool isExternal_;

public:
  USING_EXPR_BASE_NAMES

  /// @name Constructor & Destructor
  /// @{
  VarAccessExpr(const std::string& name, std::shared_ptr<Expr<DataTraits>> index = nullptr,
                SourceLocation loc = SourceLocation());
  VarAccessExpr(const VarAccessExpr<DataTraits>& expr);
  VarAccessExpr<DataTraits>& operator=(VarAccessExpr<DataTraits> expr);
  virtual ~VarAccessExpr();
  /// @}

  const std::string& getName() const { return name_; }

  void setName(std::string name) { name_ = name; }

  void setIsExternal(bool external) { isExternal_ = external; }

  /// @brief Is the variable externally defined (e.g access to a global)?
  bool isExternal() const { return isExternal_; }

  /// @brief Is is local varible access?
  bool isLocal() const { return !isExternal(); }

  /// @brief Is it an array access (i.e var[i])?
  bool isArrayAccess() const { return index_ != nullptr; }
  const std::shared_ptr<Expr<DataTraits>>& getIndex() const { return index_; }
  void setIndex(const std::shared_ptr<Expr<DataTraits>>& index) { index_ = index; }

  virtual std::shared_ptr<Expr<DataTraits>> clone() const override;
  virtual bool equals(const Expr<DataTraits>* other) const override;
  static bool classof(const Expr<DataTraits>* expr) { return expr->getKind() == EK_VarAccessExpr; }
  virtual ExprRangeType getChildren() override {
    return (isArrayAccess() ? ExprRangeType(index_) : ExprRangeType());
  }
  virtual void replaceChildren(const std::shared_ptr<Expr<DataTraits>>& oldExpr,
                               const std::shared_ptr<Expr<DataTraits>>& newExpr) override;
  ACCEPTVISITOR(Expr<DataTraits>, VarAccessExpr<DataTraits>)
};

//===------------------------------------------------------------------------------------------===//
//     FieldAccessExpr
//===------------------------------------------------------------------------------------------===//

/// @brief Field access expression
/// @ingroup ast
template <typename DataTraits>
class FieldAccessExpr : public DataTraits::FieldAccessExpr, public Expr<DataTraits> {
  std::string name_;

  // The offset known so far. If we have directional or offset arguments, we have to perform a
  // lazy evaluation to compute the real offset once we know the mapping of the directions (and
  // offsets) to the actual arguments of the stencil function.
  Array3i offset_;

  // Mapping of the directional and offset arguments of the stencil function.
  // The `argumentMap` stores an index to the argument list of the stencil function with -1
  // indicating this argument is unused. The `argumentOffset` holds the parsed offsets of the
  // direction (or offset).
  //
  // Consider the following example (given in the gridtools_clang DSL) which implements an average
  // stencil function :
  //
  // stencil_function avg {
  //   storage in;
  //   direction dir;
  //
  //   Do {
  //    return in(dir+1) + in;
  //   }
  // };
  //
  // The `in(dir+1)` FieldAccess would have the following members:
  //  - name_             : "in"
  //  - offset_           : {0, 0, 0}         // We don't have any i,j or k accesses
  //  - argumentMap_      : {1, -1, -1}       // `dir` maps to the 1st argument of `avg` (0 based)
  //  - argumentOffset_   : {1, 0, 0}         // `dir+1` has an offset `+1`
  //
  Array3i argumentMap_;
  Array3i argumentOffset_;

  // Negate the offset (this allows writing `in(-off)`)
  bool negateOffset_;

public:
  USING_EXPR_BASE_NAMES

  /// @name Constructor & Destructor
  /// @{
  FieldAccessExpr(const std::string& name, Array3i offset = Array3i{{0, 0, 0}},
                  Array3i argumentMap = Array3i{{-1, -1, -1}},
                  Array3i argumentOffset = Array3i{{0, 0, 0}}, bool negateOffset = false,
                  SourceLocation loc = SourceLocation());
  FieldAccessExpr(const FieldAccessExpr<DataTraits>& expr);
  FieldAccessExpr<DataTraits>& operator=(FieldAccessExpr<DataTraits> expr);
  virtual ~FieldAccessExpr();
  /// @}

  /// @brief Check if the fields references any arguments
  bool hasArguments() const {
    return (argumentMap_[0] != -1 || argumentMap_[1] != -1 || argumentMap_[2] != -1);
  }

  /// @brief Set the `offset` and reset the argument and argument-offset maps
  ///
  /// This function is used during the inlining when we now all the offsets.
  void setPureOffset(const Array3i& offset);

  const std::string& getName() const { return name_; }

  void setName(std::string name) { name_ = name; }

  const Array3i& getOffset() const { return offset_; }
  Array3i& getOffset() { return offset_; }

  const Array3i& getArgumentMap() const { return argumentMap_; }
  Array3i& getArgumentMap() { return argumentMap_; }

  const Array3i& getArgumentOffset() const { return argumentOffset_; }
  Array3i& getArgumentOffset() { return argumentOffset_; }

  bool negateOffset() const { return negateOffset_; }

  void setArgumentMap(Array3i const& argMap) { argumentMap_ = argMap; }

  void setArgumentOffset(Array3i const& argOffset) { argumentOffset_ = argOffset; }

  virtual std::shared_ptr<Expr<DataTraits>> clone() const override;
  virtual bool equals(const Expr<DataTraits>* other) const override;
  static bool classof(const Expr<DataTraits>* expr) {
    return expr->getKind() == EK_FieldAccessExpr;
  }
  ACCEPTVISITOR(Expr<DataTraits>, FieldAccessExpr<DataTraits>)
};

//===------------------------------------------------------------------------------------------===//
//     LiteralAccessExpr
//===------------------------------------------------------------------------------------------===//

/// @brief Variable access expression
/// @ingroup ast
template <typename DataTraits>
class LiteralAccessExpr : public DataTraits::LiteralAccessExpr, public Expr<DataTraits> {
  std::string value_;
  BuiltinTypeID builtinType_;

public:
  USING_EXPR_BASE_NAMES

  /// @name Constructor & Destructor
  /// @{
  LiteralAccessExpr(const std::string& value, BuiltinTypeID builtinType,
                    SourceLocation loc = SourceLocation());
  LiteralAccessExpr(const LiteralAccessExpr<DataTraits>& expr);
  LiteralAccessExpr<DataTraits>& operator=(LiteralAccessExpr<DataTraits> expr);
  virtual ~LiteralAccessExpr();
  /// @}

  const std::string& getValue() const { return value_; }
  std::string& getValue() { return value_; }

  const BuiltinTypeID& getBuiltinType() const { return builtinType_; }
  BuiltinTypeID& getBuiltinType() { return builtinType_; }

  virtual std::shared_ptr<Expr<DataTraits>> clone() const override;
  virtual bool equals(const Expr<DataTraits>* other) const override;
  static bool classof(const Expr<DataTraits>* expr) {
    return expr->getKind() == EK_LiteralAccessExpr;
  }
  ACCEPTVISITOR(Expr<DataTraits>, LiteralAccessExpr<DataTraits>)
};
} // namespace ast
} // namespace dawn

#include "dawn/AST/ASTExpr.tcc"

#endif
