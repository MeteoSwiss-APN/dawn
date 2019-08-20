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

#ifndef DAWN_SIR_ASTEXPR_H
#define DAWN_SIR_ASTEXPR_H

#include "dawn/Support/Array.h"
#include "dawn/Support/ArrayRef.h"
#include "dawn/Support/SourceLocation.h"
#include "dawn/Support/Type.h"
#include "dawn/Support/UIDGenerator.h"
#include "dawn/Support/VisitorHelpers.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dawn {

class ASTVisitor;

/// @brief Abstract base class of all expressions
/// @ingroup sir
class Expr : public std::enable_shared_from_this<Expr> {
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

  using ExprRangeType = ArrayRef<std::shared_ptr<Expr>>;

  /// @name Constructor & Destructor
  /// @{
  Expr(ExprKind kind, SourceLocation loc = SourceLocation())
      : kind_(kind), loc_(loc), expressionID_(UIDGenerator::getInstance()->get()) {}
  virtual ~Expr() {}
  /// @}

  /// @brief Hook for Visitors
  virtual void accept(ASTVisitor& visitor) = 0;
  virtual void accept(ASTVisitorNonConst& visitor) = 0;
  virtual std::shared_ptr<Expr> acceptAndReplace(ASTVisitorPostOrder& visitor) = 0;

  /// @brief Clone the current expression
  virtual std::shared_ptr<Expr> clone() const = 0;

  /// @brief Get kind of Expr (used by RTTI dyn_cast<> et al.)
  ExprKind getKind() const { return kind_; }

  /// @brief Get original source location
  const SourceLocation& getSourceLocation() const { return loc_; }

  /// @brief Iterate children (if any)
  virtual ExprRangeType getChildren() { return ExprRangeType(); }

  virtual void replaceChildren(const std::shared_ptr<Expr>& old_,
                               const std::shared_ptr<Expr>& new_) {
    DAWN_ASSERT(false);
  }

  /// @brief Compare for equality
  /// @{
  virtual bool equals(const std::shared_ptr<Expr>& other) const { return equals(other.get()); }
  virtual bool equals(const Expr* other) const { return kind_ == other->kind_; }
  /// @}

  /// @name Operators
  /// @{
  bool operator==(const Expr& other) const { return other.equals(this); }
  bool operator!=(const Expr& other) const { return !(*this == other); }
  /// @}

  /// @brief get the expressionID for mapping
  int getID() const { return expressionID_; }

  void setID(int id) { expressionID_ = id; }

protected:
  void assign(const Expr& other) {
    kind_ = other.kind_;
    loc_ = other.loc_;
  }

protected:
  ExprKind kind_;
  SourceLocation loc_;

  int expressionID_;
};

//===------------------------------------------------------------------------------------------===//
//     UnaryOperator
//===------------------------------------------------------------------------------------------===//

/// @brief Unary Operations (i.e `op operand`)
/// @ingroup sir
class UnaryOperator : public Expr {
protected:
  std::shared_ptr<Expr> operand_;
  std::string op_;

public:
  /// @name Constructor & Destructor
  /// @{
  UnaryOperator(const std::shared_ptr<Expr>& operand, std::string op,
                SourceLocation loc = SourceLocation());
  UnaryOperator(const UnaryOperator& expr);
  UnaryOperator& operator=(UnaryOperator expr);
  virtual ~UnaryOperator();
  /// @}

  void setOperand(const std::shared_ptr<Expr>& operand) { operand_ = operand; }
  const std::shared_ptr<Expr>& getOperand() const { return operand_; }
  const char* getOp() const { return op_.c_str(); }

  virtual std::shared_ptr<Expr> clone() const override;
  virtual bool equals(const Expr* other) const override;
  static bool classof(const Expr* expr) { return expr->getKind() == EK_UnaryOperator; }
  virtual ExprRangeType getChildren() override { return ExprRangeType(operand_); }
  virtual void replaceChildren(const std::shared_ptr<Expr>& oldExpr,
                               const std::shared_ptr<Expr>& newExpr) override;
  ACCEPTVISITOR(Expr, UnaryOperator)
};

//===------------------------------------------------------------------------------------------===//
//     BinaryOperator
//===------------------------------------------------------------------------------------------===//

/// @brief Binary Operations (i.e `left op right`)
/// @ingroup sir
class BinaryOperator : public Expr {
protected:
  enum OperandKind { OK_Left = 0, OK_Right };
  std::array<std::shared_ptr<Expr>, 2> operands_;
  std::string op_;

public:
  /// @name Constructor & Destructor
  /// @{
  BinaryOperator(const std::shared_ptr<Expr>& left, std::string op,
                 const std::shared_ptr<Expr>& right, SourceLocation loc = SourceLocation());
  BinaryOperator(const BinaryOperator& expr);
  BinaryOperator& operator=(BinaryOperator expr);
  virtual ~BinaryOperator();
  /// @}

  void setLeft(const std::shared_ptr<Expr>& left) { operands_[OK_Left] = left; }
  const std::shared_ptr<Expr>& getLeft() const { return operands_[OK_Left]; }
  std::shared_ptr<Expr>& getLeft() { return operands_[OK_Left]; }

  void setRight(const std::shared_ptr<Expr>& right) { operands_[OK_Right] = right; }
  const std::shared_ptr<Expr>& getRight() const { return operands_[OK_Right]; }
  std::shared_ptr<Expr>& getRight() { return operands_[OK_Right]; }

  const char* getOp() const { return op_.c_str(); }

  virtual std::shared_ptr<Expr> clone() const override;
  virtual bool equals(const Expr* other) const override;
  static bool classof(const Expr* expr) { return expr->getKind() == EK_BinaryOperator; }
  virtual ExprRangeType getChildren() override { return ExprRangeType(operands_); }
  virtual void replaceChildren(const std::shared_ptr<Expr>& oldExpr,
                               const std::shared_ptr<Expr>& newExpr) override;

  ACCEPTVISITOR(Expr, BinaryOperator)
};

//===------------------------------------------------------------------------------------------===//
//     AssignmentExpr
//===------------------------------------------------------------------------------------------===//

/// @brief Assignment expression (i.e `left = right`)
/// @ingroup sir
class AssignmentExpr : public BinaryOperator {
public:
  /// @name Constructor & Destructor
  /// @{
  AssignmentExpr(const std::shared_ptr<Expr>& left, const std::shared_ptr<Expr>& right,
                 std::string op = "=", SourceLocation loc = SourceLocation());
  AssignmentExpr(const AssignmentExpr& expr);
  AssignmentExpr& operator=(AssignmentExpr expr);
  virtual ~AssignmentExpr();
  /// @}

  virtual std::shared_ptr<Expr> clone() const override;
  virtual bool equals(const Expr* other) const override;
  static bool classof(const Expr* expr) { return expr->getKind() == EK_AssignmentExpr; }
  ACCEPTVISITOR(Expr, AssignmentExpr)
};

//===------------------------------------------------------------------------------------------===//
//     AssignmentExpr
//===------------------------------------------------------------------------------------------===//

/// @brief NOP expression
/// @ingroup sir
class NOPExpr : public Expr {
public:
  /// @name Constructor & Destructor
  /// @{
  NOPExpr(SourceLocation loc = SourceLocation());
  NOPExpr(const NOPExpr& expr);
  NOPExpr& operator=(NOPExpr expr);
  virtual ~NOPExpr();
  /// @}

  virtual std::shared_ptr<Expr> clone() const override;
  virtual bool equals(const Expr* other) const override;
  static bool classof(const Expr* expr) { return expr->getKind() == EK_NOPExpr; }
  ACCEPTVISITOR(Expr, NOPExpr)
};

//===------------------------------------------------------------------------------------------===//
//     TernaryOperator
//===------------------------------------------------------------------------------------------===//

/// @brief Ternary Operations (i.e `condition ? left : right`)
/// @ingroup sir
class TernaryOperator : public Expr {
protected:
  enum OperandKind { OK_Cond = 0, OK_Left, OK_Right };
  std::array<std::shared_ptr<Expr>, 3> operands_;

public:
  /// @name Constructor & Destructor
  /// @{
  TernaryOperator(const std::shared_ptr<Expr>& cond, const std::shared_ptr<Expr>& left,
                  const std::shared_ptr<Expr>& right, SourceLocation loc = SourceLocation());
  TernaryOperator(const TernaryOperator& expr);
  TernaryOperator& operator=(TernaryOperator expr);
  virtual ~TernaryOperator();
  /// @}

  void setCondition(const std::shared_ptr<Expr>& condition) { operands_[OK_Cond] = condition; }
  const std::shared_ptr<Expr>& getCondition() const { return operands_[OK_Cond]; }
  std::shared_ptr<Expr>& getCondition() { return operands_[OK_Cond]; }

  void setLeft(const std::shared_ptr<Expr>& left) { operands_[OK_Left] = left; }
  const std::shared_ptr<Expr>& getLeft() const { return operands_[OK_Left]; }
  std::shared_ptr<Expr>& getLeft() { return operands_[OK_Left]; }

  void setRight(const std::shared_ptr<Expr>& right) { operands_[OK_Right] = right; }
  const std::shared_ptr<Expr>& getRight() const { return operands_[OK_Right]; }
  std::shared_ptr<Expr>& getRight() { return operands_[OK_Right]; }

  const char* getOp() const { return "?"; }
  const char* getSeperator() const { return ":"; }

  virtual std::shared_ptr<Expr> clone() const override;
  virtual bool equals(const Expr* other) const override;
  static bool classof(const Expr* expr) { return expr->getKind() == EK_TernaryOperator; }
  virtual ExprRangeType getChildren() override { return ExprRangeType(operands_); }
  virtual void replaceChildren(const std::shared_ptr<Expr>& oldExpr,
                               const std::shared_ptr<Expr>& newExpr) override;
  ACCEPTVISITOR(Expr, TernaryOperator)
};

//===------------------------------------------------------------------------------------------===//
//     FunCallExpr
//===------------------------------------------------------------------------------------------===//

/// @brief Call expression
/// @ingroup sir
class FunCallExpr : public Expr {
protected:
  std::string callee_;
  std::vector<std::shared_ptr<Expr>> arguments_;

public:
  /// @name Constructor & Destructor
  /// @{
  FunCallExpr(const std::string& callee, SourceLocation loc = SourceLocation());
  FunCallExpr(const FunCallExpr& expr);
  FunCallExpr& operator=(FunCallExpr expr);
  virtual ~FunCallExpr();
  /// @}

  std::string& getCallee() { return callee_; }
  const std::string& getCallee() const { return callee_; }

  std::vector<std::shared_ptr<Expr>>& getArguments() { return arguments_; }
  const std::vector<std::shared_ptr<Expr>>& getArguments() const { return arguments_; }

  void setCallee(std::string name) { callee_ = name; }

  void insertArgument(const std::shared_ptr<Expr>& expr);

  template <typename Iterator>
  inline void insertArguments(Iterator begin, Iterator end) {
    for(Iterator it = begin; it != end; ++it) {
      insertArgument(*it);
    }
  }

  virtual std::shared_ptr<Expr> clone() const override;
  virtual bool equals(const Expr* other) const override;
  static bool classof(const Expr* expr) { return expr->getKind() == EK_FunCallExpr; }
  virtual ExprRangeType getChildren() override { return ExprRangeType(arguments_); }
  virtual void replaceChildren(const std::shared_ptr<Expr>& oldExpr,
                               const std::shared_ptr<Expr>& newExpr) override;
  ACCEPTVISITOR(Expr, FunCallExpr)
};

//===------------------------------------------------------------------------------------------===//
//     StencilFunCallExpr
//===------------------------------------------------------------------------------------------===//

/// @brief Stencil function call
/// @ingroup sir
class StencilFunCallExpr : public FunCallExpr {
public:
  /// @name Constructor & Destructor
  /// @{
  StencilFunCallExpr(const std::string& callee, SourceLocation loc = SourceLocation());
  StencilFunCallExpr(const StencilFunCallExpr& expr);
  StencilFunCallExpr& operator=(StencilFunCallExpr expr);
  virtual ~StencilFunCallExpr();
  /// @}

  //  void setName(std::string name);
  virtual std::shared_ptr<Expr> clone() const override;
  virtual bool equals(const Expr* other) const override;
  static bool classof(const Expr* expr) { return expr->getKind() == EK_StencilFunCallExpr; }
  ACCEPTVISITOR(Expr, StencilFunCallExpr)
};

//===------------------------------------------------------------------------------------------===//
//     StencilFunArgExpr
//===------------------------------------------------------------------------------------------===//

/// @brief Stencil function argument (direction or offset)
/// @ingroup sir
class StencilFunArgExpr : public Expr {
  int dimension_;
  int offset_;
  int argumentIndex_;

public:
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
  StencilFunArgExpr(const StencilFunArgExpr& expr);
  StencilFunArgExpr& operator=(StencilFunArgExpr expr);
  virtual ~StencilFunArgExpr();
  /// @}

  bool needsLazyEval() const { return argumentIndex_ != -1; }

  int getDimension() const { return dimension_; }
  int getOffset() const { return offset_; }
  int getArgumentIndex() const { return argumentIndex_; }

  virtual std::shared_ptr<Expr> clone() const override;
  virtual bool equals(const Expr* other) const override;
  static bool classof(const Expr* expr) { return expr->getKind() == EK_StencilFunArgExpr; }
  ACCEPTVISITOR(Expr, StencilFunArgExpr)
};

//===------------------------------------------------------------------------------------------===//
//     VarAccessExpr
//===------------------------------------------------------------------------------------------===//

/// @brief Variable access expression
/// @ingroup sir
class VarAccessExpr : public Expr {
  std::string name_;
  std::shared_ptr<Expr> index_;
  bool isExternal_;

public:
  /// @name Constructor & Destructor
  /// @{
  VarAccessExpr(const std::string& name, std::shared_ptr<Expr> index = nullptr,
                SourceLocation loc = SourceLocation());
  VarAccessExpr(const VarAccessExpr& expr);
  VarAccessExpr& operator=(VarAccessExpr expr);
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
  const std::shared_ptr<Expr>& getIndex() const { return index_; }
  void setIndex(const std::shared_ptr<Expr>& index) { index_ = index; }

  virtual std::shared_ptr<Expr> clone() const override;
  virtual bool equals(const Expr* other) const override;
  static bool classof(const Expr* expr) { return expr->getKind() == EK_VarAccessExpr; }
  virtual ExprRangeType getChildren() override {
    return (isArrayAccess() ? ExprRangeType(index_) : ExprRangeType());
  }
  virtual void replaceChildren(const std::shared_ptr<Expr>& oldExpr,
                               const std::shared_ptr<Expr>& newExpr) override;
  ACCEPTVISITOR(Expr, VarAccessExpr)
};

//===------------------------------------------------------------------------------------------===//
//     FieldAccessExpr
//===------------------------------------------------------------------------------------------===//

/// @brief Field access expression
/// @ingroup sir
class FieldAccessExpr : public Expr {
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
  /// @name Constructor & Destructor
  /// @{
  FieldAccessExpr(const std::string& name, Array3i offset = Array3i{{0, 0, 0}},
                  Array3i argumentMap = Array3i{{-1, -1, -1}},
                  Array3i argumentOffset = Array3i{{0, 0, 0}}, bool negateOffset = false,
                  SourceLocation loc = SourceLocation());
  FieldAccessExpr(const FieldAccessExpr& expr);
  FieldAccessExpr& operator=(FieldAccessExpr expr);
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

  virtual std::shared_ptr<Expr> clone() const override;
  virtual bool equals(const Expr* other) const override;
  static bool classof(const Expr* expr) { return expr->getKind() == EK_FieldAccessExpr; }
  ACCEPTVISITOR(Expr, FieldAccessExpr)
};

//===------------------------------------------------------------------------------------------===//
//     LiteralAccessExpr
//===------------------------------------------------------------------------------------------===//

/// @brief Variable access expression
/// @ingroup sir
class LiteralAccessExpr : public Expr {
  std::string value_;
  BuiltinTypeID builtinType_;

public:
  /// @name Constructor & Destructor
  /// @{
  LiteralAccessExpr(const std::string& value, BuiltinTypeID builtinType,
                    SourceLocation loc = SourceLocation());
  LiteralAccessExpr(const LiteralAccessExpr& expr);
  LiteralAccessExpr& operator=(LiteralAccessExpr expr);
  virtual ~LiteralAccessExpr();
  /// @}

  const std::string& getValue() const { return value_; }
  std::string& getValue() { return value_; }

  const BuiltinTypeID& getBuiltinType() const { return builtinType_; }
  BuiltinTypeID& getBuiltinType() { return builtinType_; }

  virtual std::shared_ptr<Expr> clone() const override;
  virtual bool equals(const Expr* other) const override;
  static bool classof(const Expr* expr) { return expr->getKind() == EK_LiteralAccessExpr; }
  ACCEPTVISITOR(Expr, LiteralAccessExpr)
};

} // namespace dawn

#endif
