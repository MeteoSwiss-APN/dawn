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

#ifndef DAWN_SIR_ASTSTMT_H
#define DAWN_SIR_ASTSTMT_H

#include "dawn/Support/ArrayRef.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/SourceLocation.h"
#include "dawn/Support/Type.h"
#include <memory>
#include <vector>

namespace dawn {

namespace sir {
struct VerticalRegion;
struct StencilCall;
struct Field;
}

class ASTVisitor;
class Expr;

/// @brief Abstract base class of all statements
/// @ingroup sir
class Stmt : public std::enable_shared_from_this<Stmt> {
public:
  /// @brief Discriminator for RTTI (dyn_cast<> et al.)
  enum StmtKind {
    SK_BlockStmt,
    SK_ExprStmt,
    SK_ReturnStmt,
    SK_VarDeclStmt,
    SK_StencilCallDeclStmt,
    SK_VerticalRegionDeclStmt,
    SK_BoundaryConditionDeclStmt,
    SK_IfStmt
  };

  using StmtRangeType = MutableArrayRef<std::shared_ptr<Stmt>>;

  /// @name Constructor & Destructor
  /// @{
  Stmt(StmtKind kind, SourceLocation loc = SourceLocation()) : kind_(kind), loc_(loc) {}
  virtual ~Stmt() {}
  /// @}

  /// @brief Hook for Visitors
  virtual void accept(ASTVisitor& visitor) = 0;

  /// @brief Clone the current statement
  virtual std::shared_ptr<Stmt> clone() const = 0;

  /// @brief Get kind of Stmt (used by RTTI dyn_cast<> et al.)
  StmtKind getKind() const { return kind_; }

  /// @brief Get original source location
  const SourceLocation& getSourceLocation() const { return loc_; }

  /// @brief Iterate children (if any)
  virtual StmtRangeType getChildren() { return StmtRangeType(); }

  /// @brief Compare for equality
  virtual bool equals(const Stmt* other) const { return kind_ == other->kind_; }

  /// @brief Is the statement used for stencil description and has no real analogon in C++
  /// (e.g a VerticalRegion or StencilCall)?
  virtual bool isStencilDesc() const { return false; }

  /// @name Operators
  /// @{
  bool operator==(const Stmt& other) const { return other.equals(this); }
  bool operator!=(const Stmt& other) const { return !(*this == other); }
  /// @}

protected:
  void assign(const Stmt& other) {
    kind_ = other.kind_;
    loc_ = other.loc_;
  }

protected:
  StmtKind kind_;
  SourceLocation loc_;
};

//===------------------------------------------------------------------------------------------===//
//     BlockStmt
//===------------------------------------------------------------------------------------------===//

/// @brief Block of statements
/// @ingroup sir
class BlockStmt : public Stmt {
  std::vector<std::shared_ptr<Stmt>> statements_;

public:
  using StatementList = std::vector<std::shared_ptr<Stmt>>;

  /// @name Constructor & Destructor
  /// @{
  BlockStmt(SourceLocation loc = SourceLocation());
  BlockStmt(const std::vector<std::shared_ptr<Stmt>>& statements,
            SourceLocation loc = SourceLocation());
  BlockStmt(const BlockStmt& stmt);
  BlockStmt& operator=(BlockStmt stmt);
  virtual ~BlockStmt();
  /// @}

  template <class Range>
  void insert_back(Range&& range) {
    insert_back(std::begin(range), std::end(range));
  }

  template <class Iterator>
  void insert_back(Iterator begin, Iterator end) {
    statements_.insert(statements_.end(), begin, end);
  }

  void push_back(const std::shared_ptr<Stmt>& stmt) { statements_.push_back(stmt); }

  std::vector<std::shared_ptr<Stmt>>& getStatements() { return statements_; }
  const std::vector<std::shared_ptr<Stmt>>& getStatements() const { return statements_; }

  virtual std::shared_ptr<Stmt> clone() const override;
  virtual bool equals(const Stmt* other) const override;
  virtual void accept(ASTVisitor& visitor) override;
  static bool classof(const Stmt* stmt) { return stmt->getKind() == SK_BlockStmt; }
  virtual StmtRangeType getChildren() override { return StmtRangeType(statements_); }
};

//===------------------------------------------------------------------------------------------===//
//     ExprStmt
//===------------------------------------------------------------------------------------------===//

/// @brief Block of statements
/// @ingroup sir
class ExprStmt : public Stmt {
  std::shared_ptr<Expr> expr_;

public:
  /// @name Constructor & Destructor
  /// @{
  ExprStmt(const std::shared_ptr<Expr>& expr, SourceLocation loc = SourceLocation());
  ExprStmt(const ExprStmt& stmt);
  ExprStmt& operator=(ExprStmt stmt);
  virtual ~ExprStmt();
  /// @}

  void setExpr(const std::shared_ptr<Expr>& expr) { expr_ = expr; }
  const std::shared_ptr<Expr>& getExpr() const { return expr_; }
  std::shared_ptr<Expr>& getExpr() { return expr_; }

  virtual std::shared_ptr<Stmt> clone() const override;
  virtual bool equals(const Stmt* other) const override;
  virtual void accept(ASTVisitor& visitor) override;
  static bool classof(const Stmt* stmt) { return stmt->getKind() == SK_ExprStmt; }
};

//===------------------------------------------------------------------------------------------===//
//     ReturnStmt
//===------------------------------------------------------------------------------------------===//

/// @brief This represents a return of an expression
/// @ingroup sir
class ReturnStmt : public Stmt {
  std::shared_ptr<Expr> expr_;

public:
  /// @name Constructor & Destructor
  /// @{
  ReturnStmt(const std::shared_ptr<Expr>& expr, SourceLocation loc = SourceLocation());
  ReturnStmt(const ReturnStmt& stmt);
  ReturnStmt& operator=(ReturnStmt stmt);
  virtual ~ReturnStmt();
  /// @}

  void setExpr(const std::shared_ptr<Expr>& expr) { expr_ = expr; }
  const std::shared_ptr<Expr>& getExpr() const { return expr_; }
  std::shared_ptr<Expr>& getExpr() { return expr_; }

  virtual std::shared_ptr<Stmt> clone() const override;
  virtual bool equals(const Stmt* other) const override;
  virtual void accept(ASTVisitor& visitor) override;
  static bool classof(const Stmt* stmt) { return stmt->getKind() == SK_ReturnStmt; }
};

//===------------------------------------------------------------------------------------------===//
//     VarDeclStmt
//===------------------------------------------------------------------------------------------===//

/// @brief This represents a declaration of a local variable or C-array
/// @ingroup sir
class VarDeclStmt : public Stmt {
  Type type_;
  std::string name_;

  // Dimension of the array or 0 for variables
  int dimension_;
  const char* op_;

  // List of expression used for initializaion or just 1 element for variables
  std::vector<std::shared_ptr<Expr>> initList_;

public:
  /// @name Constructor & Destructor
  /// @{
  VarDeclStmt(const Type& type, const std::string& name, int dimension, const char* op,
              std::vector<std::shared_ptr<Expr>> initList, SourceLocation loc = SourceLocation());
  VarDeclStmt(const VarDeclStmt& stmt);
  VarDeclStmt& operator=(VarDeclStmt stmt);
  virtual ~VarDeclStmt();
  /// @}

  const Type& getType() const { return type_; }
  Type& getType() { return type_; }

  const std::string& getName() const { return name_; }
  std::string& getName() { return name_; }

  const char* getOp() const { return op_; }
  int getDimension() const { return dimension_; }

  bool isArray() const { return (dimension_ > 0); }
  bool hasInit() const { return (!initList_.empty()); }
  const std::vector<std::shared_ptr<Expr>>& getInitList() const { return initList_; }
  std::vector<std::shared_ptr<Expr>>& getInitList() { return initList_; }

  virtual std::shared_ptr<Stmt> clone() const override;
  virtual bool equals(const Stmt* other) const override;
  virtual void accept(ASTVisitor& visitor) override;
  static bool classof(const Stmt* stmt) { return stmt->getKind() == SK_VarDeclStmt; }
};

//===------------------------------------------------------------------------------------------===//
//     VerticalRegionDeclStmt
//===------------------------------------------------------------------------------------------===//

/// @brief This represents a declaration of a sir::VerticalRegion
/// @ingroup sir
class VerticalRegionDeclStmt : public Stmt {
  std::shared_ptr<sir::VerticalRegion> verticalRegion_;

public:
  /// @name Constructor & Destructor
  /// @{
  VerticalRegionDeclStmt(const std::shared_ptr<sir::VerticalRegion>& verticalRegion,
                         SourceLocation loc = SourceLocation());
  VerticalRegionDeclStmt(const VerticalRegionDeclStmt& stmt);
  VerticalRegionDeclStmt& operator=(VerticalRegionDeclStmt stmt);
  virtual ~VerticalRegionDeclStmt();
  /// @}

  const std::shared_ptr<sir::VerticalRegion>& getVerticalRegion() const { return verticalRegion_; }

  virtual bool isStencilDesc() const override { return true; }
  virtual std::shared_ptr<Stmt> clone() const override;
  virtual bool equals(const Stmt* other) const override;
  virtual void accept(ASTVisitor& visitor) override;
  static bool classof(const Stmt* stmt) { return stmt->getKind() == SK_VerticalRegionDeclStmt; }
};

//===------------------------------------------------------------------------------------------===//
//     StencilCallDeclStmt
//===------------------------------------------------------------------------------------------===//

/// @brief This represents a declaration of a sir::StencilCall
/// @ingroup sir
class StencilCallDeclStmt : public Stmt {
  std::shared_ptr<sir::StencilCall> stencilCall_;

public:
  /// @name Constructor & Destructor
  /// @{
  StencilCallDeclStmt(const std::shared_ptr<sir::StencilCall>& stencilCall,
                      SourceLocation loc = SourceLocation());
  StencilCallDeclStmt(const StencilCallDeclStmt& stmt);
  StencilCallDeclStmt& operator=(StencilCallDeclStmt stmt);
  virtual ~StencilCallDeclStmt();
  /// @}

  const std::shared_ptr<sir::StencilCall>& getStencilCall() const { return stencilCall_; }

  virtual bool isStencilDesc() const override { return true; }
  virtual std::shared_ptr<Stmt> clone() const override;
  virtual bool equals(const Stmt* other) const override;
  virtual void accept(ASTVisitor& visitor) override;
  static bool classof(const Stmt* stmt) { return stmt->getKind() == SK_StencilCallDeclStmt; }
};

//===------------------------------------------------------------------------------------------===//
//     BoundaryConditionDeclStmt
//===------------------------------------------------------------------------------------------===//

/// @brief This represents a declaration of a boundary condition
/// @ingroup sir
class BoundaryConditionDeclStmt : public Stmt {
  std::string functor_;
  std::vector<std::shared_ptr<sir::Field>> fields_;

public:
  /// @name Constructor & Destructor
  /// @{
  BoundaryConditionDeclStmt(const std::string& callee, SourceLocation loc = SourceLocation());
  BoundaryConditionDeclStmt(const BoundaryConditionDeclStmt& stmt);
  BoundaryConditionDeclStmt& operator=(BoundaryConditionDeclStmt stmt);
  virtual ~BoundaryConditionDeclStmt();
  /// @}

  const std::string& getFunctor() const { return functor_; }

  std::vector<std::shared_ptr<sir::Field>>& getFields() { return fields_; }
  const std::vector<std::shared_ptr<sir::Field>>& getFields() const { return fields_; }

  virtual bool isStencilDesc() const override { return true; }
  virtual std::shared_ptr<Stmt> clone() const override;
  virtual bool equals(const Stmt* other) const override;
  virtual void accept(ASTVisitor& visitor) override;
  static bool classof(const Stmt* stmt) { return stmt->getKind() == SK_BoundaryConditionDeclStmt; }
};

//===------------------------------------------------------------------------------------------===//
//     IfStmt
//===------------------------------------------------------------------------------------------===//

/// @brief This represents an if/then/else block
/// @ingroup sir
class IfStmt : public Stmt {
  enum OperandKind { OK_Cond, OK_Then, OK_Else, OK_End };
  std::shared_ptr<Stmt> subStmts_[OK_End];

public:
  /// @name Constructor & Destructor
  /// @{
  IfStmt(const std::shared_ptr<Stmt>& condExpr, const std::shared_ptr<Stmt>& thenStmt,
         const std::shared_ptr<Stmt>& elseStmt = nullptr, SourceLocation loc = SourceLocation());
  IfStmt(const IfStmt& stmt);
  IfStmt& operator=(IfStmt stmt);
  virtual ~IfStmt();
  /// @}

  const std::shared_ptr<Expr>& getCondExpr() const {
    return dyn_cast<ExprStmt>(subStmts_[OK_Cond].get())->getExpr();
  }
  std::shared_ptr<Expr>& getCondExpr() {
    return dyn_cast<ExprStmt>(subStmts_[OK_Cond].get())->getExpr();
  }

  const std::shared_ptr<Stmt>& getCondStmt() const { return subStmts_[OK_Cond]; }
  std::shared_ptr<Stmt>& getCondStmt() { return subStmts_[OK_Cond]; }

  const std::shared_ptr<Stmt>& getThenStmt() const { return subStmts_[OK_Then]; }
  std::shared_ptr<Stmt>& getThenStmt() { return subStmts_[OK_Then]; }
  void setThenStmt(std::shared_ptr<Stmt>& thenStmt) { subStmts_[OK_Then] = thenStmt; }

  const std::shared_ptr<Stmt>& getElseStmt() const { return subStmts_[OK_Else]; }
  std::shared_ptr<Stmt>& getElseStmt() { return subStmts_[OK_Else]; }
  bool hasElse() const { return getElseStmt() != nullptr; }
  void setElseStmt(std::shared_ptr<Stmt>& elseStmt) { subStmts_[OK_Else] = elseStmt; }

  virtual std::shared_ptr<Stmt> clone() const override;
  virtual bool equals(const Stmt* other) const override;
  virtual void accept(ASTVisitor& visitor) override;
  static bool classof(const Stmt* stmt) { return stmt->getKind() == SK_IfStmt; }
  virtual StmtRangeType getChildren() override {
    return hasElse() ? StmtRangeType(subStmts_) : StmtRangeType(&subStmts_[0], OK_End - 1);
  }
};

} // namespace dawn

#endif
