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

#ifndef DAWN_AST_ASTSTMT_H
#define DAWN_AST_ASTSTMT_H

#include "dawn/AST/ASTVisitorHelpers.h"
#include "dawn/Support/ArrayRef.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/ComparisonHelpers.h"
#include "dawn/Support/SourceLocation.h"
#include "dawn/Support/Type.h"
#include "dawn/Support/UIDGenerator.h"
#include <memory>
#include <vector>

namespace dawn {

namespace ast {
template <typename DataTraits>
class AST;
template <typename DataTraits>
class Expr;

/// @brief Abstract base class of all statements
/// @ingroup ast
template <typename DataTraits>
class Stmt : virtual public DataTraits::StmtData,
             public std::enable_shared_from_this<Stmt<DataTraits>> {
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

  using StmtRangeType = MutableArrayRef<std::shared_ptr<Stmt<DataTraits>>>;

  /// @name Constructor & Destructor
  /// @{
  Stmt(StmtKind kind, SourceLocation loc = SourceLocation())
      : kind_(kind), loc_(loc), statementID_(UIDGenerator::getInstance()->get()) {}
  virtual ~Stmt() {}
  /// @}

  /// @brief Hook for Visitors
  virtual void accept(ASTVisitor<DataTraits>& visitor) = 0;
  virtual void accept(ASTVisitorNonConst<DataTraits>& visitor) = 0;
  virtual void accept(ASTVisitorForwarding<DataTraits>& visitor) = 0;
  virtual std::shared_ptr<Stmt<DataTraits>>
  acceptAndReplace(ASTVisitorPostOrder<DataTraits>& visitor) = 0;
  virtual void accept(ASTVisitorForwardingNonConst<DataTraits>& visitor) = 0;
  virtual void accept(ASTVisitorDisabled<DataTraits>& visitor) = 0;

  /// @brief Clone the current statement
  virtual std::shared_ptr<Stmt<DataTraits>> clone() const = 0;

  /// @brief Get kind of Stmt (used by RTTI dyn_cast<> et al.)
  StmtKind getKind() const { return kind_; }

  /// @brief Get original source location
  const SourceLocation& getSourceLocation() const { return loc_; }
  SourceLocation& getSourceLocation() { return loc_; }

  /// @brief Iterate children (if any)
  virtual StmtRangeType getChildren() { return StmtRangeType(); }

  virtual void replaceChildren(std::shared_ptr<Stmt<DataTraits>> const& oldStmt,
                               std::shared_ptr<Stmt<DataTraits>> const& newStmt) {}

  /// @brief Compare for equality
  virtual bool equals(const Stmt<DataTraits>* other) const { return kind_ == other->kind_; }

  /// @brief Is the statement used for stencil description and has no real analogon in C++
  /// (e.g a VerticalRegion or StencilCall)?
  virtual bool isStencilDesc() const { return false; }

  /// @name Operators
  /// @{
  bool operator==(const Stmt<DataTraits>& other) const { return other.equals(this); }
  bool operator!=(const Stmt<DataTraits>& other) const { return !(*this == other); }
  /// @}

  /// @brief get the statementID for mapping
  int getID() const { return statementID_; }

  void setID(int id) { statementID_ = id; }

protected:
  void assign(const Stmt<DataTraits>& other) {
    kind_ = other.kind_;
    loc_ = other.loc_;
  }

protected:
  StmtKind kind_;
  SourceLocation loc_;
  int statementID_;
};

#define USING_STMT_BASE_NAMES                                                                      \
  using typename Stmt<DataTraits>::StmtRangeType;                                                  \
  using typename Stmt<DataTraits>::StmtKind;                                                       \
  using Stmt<DataTraits>::SK_BlockStmt;                                                            \
  using Stmt<DataTraits>::SK_ExprStmt;                                                             \
  using Stmt<DataTraits>::SK_ReturnStmt;                                                           \
  using Stmt<DataTraits>::SK_VarDeclStmt;                                                          \
  using Stmt<DataTraits>::SK_StencilCallDeclStmt;                                                  \
  using Stmt<DataTraits>::SK_VerticalRegionDeclStmt;                                               \
  using Stmt<DataTraits>::SK_BoundaryConditionDeclStmt;                                            \
  using Stmt<DataTraits>::SK_IfStmt;                                                               \
  using Stmt<DataTraits>::kind_;                                                                   \
  using Stmt<DataTraits>::loc_;                                                                    \
  using Stmt<DataTraits>::statementID_;

//===------------------------------------------------------------------------------------------===//
//     BlockStmt
//===------------------------------------------------------------------------------------------===//

/// @brief Block of statements
/// @ingroup ast
template <typename DataTraits>
class BlockStmt : public DataTraits::BlockStmt, public Stmt<DataTraits> {
public:
  USING_STMT_BASE_NAMES
  using StatementList = std::vector<std::shared_ptr<Stmt<DataTraits>>>;

  /// @name Constructor & Destructor
  /// @{
  BlockStmt(SourceLocation loc = SourceLocation());
  BlockStmt(const StatementList& statements, SourceLocation loc = SourceLocation());
  BlockStmt(const BlockStmt<DataTraits>& stmt);
  BlockStmt<DataTraits>& operator=(BlockStmt<DataTraits> const& stmt);
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

  void push_back(const std::shared_ptr<Stmt<DataTraits>>& stmt) { statements_.push_back(stmt); }

  StatementList& getStatements() { return statements_; }
  const StatementList& getStatements() const { return statements_; }

  virtual std::shared_ptr<Stmt<DataTraits>> clone() const override;
  virtual bool equals(const Stmt<DataTraits>* other) const override;
  static bool classof(const Stmt<DataTraits>* stmt) { return stmt->getKind() == SK_BlockStmt; }
  virtual StmtRangeType getChildren() override { return StmtRangeType(statements_); }
  virtual void replaceChildren(const std::shared_ptr<Stmt<DataTraits>>& oldStmt,
                               const std::shared_ptr<Stmt<DataTraits>>& newStmt) override;

  bool isEmpty() const { return statements_.empty(); }

  ACCEPTVISITOR(Stmt<DataTraits>, BlockStmt<DataTraits>)

private:
  StatementList statements_;
};

//===------------------------------------------------------------------------------------------===//
//     ExprStmt
//===------------------------------------------------------------------------------------------===//

/// @brief Block of statements
/// @ingroup ast
template <typename DataTraits>
class ExprStmt : public DataTraits::ExprStmt, public Stmt<DataTraits> {
  std::shared_ptr<Expr<DataTraits>> expr_;

public:
  USING_STMT_BASE_NAMES
  /// @name Constructor & Destructor
  /// @{
  ExprStmt(const std::shared_ptr<Expr<DataTraits>>& expr, SourceLocation loc = SourceLocation());
  ExprStmt(const ExprStmt<DataTraits>& stmt);
  ExprStmt<DataTraits>& operator=(ExprStmt<DataTraits> stmt);
  virtual ~ExprStmt();
  /// @}

  void setExpr(const std::shared_ptr<Expr<DataTraits>>& expr) { expr_ = expr; }
  const std::shared_ptr<Expr<DataTraits>>& getExpr() const { return expr_; }
  std::shared_ptr<Expr<DataTraits>>& getExpr() { return expr_; }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
  virtual void replaceChildren(const std::shared_ptr<Expr<DataTraits>>& oldExpr,
                               const std::shared_ptr<Expr<DataTraits>>& newExpr);
#pragma GCC diagnostic pop

  virtual std::shared_ptr<Stmt<DataTraits>> clone() const override;
  virtual bool equals(const Stmt<DataTraits>* other) const override;
  static bool classof(const Stmt<DataTraits>* stmt) { return stmt->getKind() == SK_ExprStmt; }
  ACCEPTVISITOR(Stmt<DataTraits>, ExprStmt<DataTraits>)
};

//===------------------------------------------------------------------------------------------===//
//     ReturnStmt
//===------------------------------------------------------------------------------------------===//

/// @brief This represents a return of an expression
/// @ingroup ast
template <typename DataTraits>
class ReturnStmt : public DataTraits::ReturnStmt, public Stmt<DataTraits> {
  std::shared_ptr<Expr<DataTraits>> expr_;

public:
  USING_STMT_BASE_NAMES
  /// @name Constructor & Destructor
  /// @{
  ReturnStmt(const std::shared_ptr<Expr<DataTraits>>& expr, SourceLocation loc = SourceLocation());
  ReturnStmt(const ReturnStmt<DataTraits>& stmt);
  ReturnStmt<DataTraits>& operator=(ReturnStmt<DataTraits> stmt);
  virtual ~ReturnStmt();
  /// @}

  void setExpr(const std::shared_ptr<Expr<DataTraits>>& expr) { expr_ = expr; }
  const std::shared_ptr<Expr<DataTraits>>& getExpr() const { return expr_; }
  std::shared_ptr<Expr<DataTraits>>& getExpr() { return expr_; }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
  virtual void replaceChildren(const std::shared_ptr<Expr<DataTraits>>& oldExpr,
                               const std::shared_ptr<Expr<DataTraits>>& newExpr);
#pragma GCC diagnostic pop

  virtual std::shared_ptr<Stmt<DataTraits>> clone() const override;
  virtual bool equals(const Stmt<DataTraits>* other) const override;
  static bool classof(const Stmt<DataTraits>* stmt) { return stmt->getKind() == SK_ReturnStmt; }
  ACCEPTVISITOR(Stmt<DataTraits>, ReturnStmt<DataTraits>)
};

//===------------------------------------------------------------------------------------------===//
//     VarDeclStmt
//===------------------------------------------------------------------------------------------===//

/// @brief This represents a declaration of a local variable or C-array
/// @ingroup ast
template <typename DataTraits>
class VarDeclStmt : public DataTraits::VarDeclStmt, public Stmt<DataTraits> {
public:
  USING_STMT_BASE_NAMES
  using InitList = std::vector<std::shared_ptr<Expr<DataTraits>>>;

  /// @name Constructor & Destructor
  /// @{
  VarDeclStmt(const Type& type, const std::string& name, int dimension, const char* op,
              InitList initList, SourceLocation loc = SourceLocation());
  VarDeclStmt(const VarDeclStmt<DataTraits>& stmt);
  VarDeclStmt<DataTraits>& operator=(VarDeclStmt<DataTraits> stmt);
  virtual ~VarDeclStmt();
  /// @}

  const Type& getType() const { return type_; }
  Type& getType() { return type_; }

  const std::string& getName() const { return name_; }
  std::string& getName() { return name_; }

  const char* getOp() const { return op_.c_str(); }
  int getDimension() const { return dimension_; }

  bool isArray() const { return (dimension_ > 0); }
  bool hasInit() const { return (!initList_.empty()); }
  const InitList& getInitList() const { return initList_; }
  InitList& getInitList() { return initList_; }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
  virtual void replaceChildren(const std::shared_ptr<Expr<DataTraits>>& oldExpr,
                               const std::shared_ptr<Expr<DataTraits>>& newExpr);
#pragma GCC diagnostic pop

  virtual std::shared_ptr<Stmt<DataTraits>> clone() const override;
  virtual bool equals(const Stmt<DataTraits>* other) const override;
  static bool classof(const Stmt<DataTraits>* stmt) { return stmt->getKind() == SK_VarDeclStmt; }
  ACCEPTVISITOR(Stmt<DataTraits>, VarDeclStmt<DataTraits>)

private:
  Type type_;
  std::string name_;

  // Dimension of the array or 0 for variables
  int dimension_;
  std::string op_;

  // List of expression used for initializaion or just 1 element for variables
  InitList initList_;
};

/// @brief Call to another stencil
/// @ingroup ast
struct StencilCall {

  SourceLocation Loc;            ///< Source location of the call
  std::string Callee;            ///< Name of the callee stencil
  std::vector<std::string> Args; ///< List of fields used as arguments

  StencilCall(std::string callee, SourceLocation loc = SourceLocation())
      : Loc(loc), Callee(callee) {}

  /// @brief Clone the vertical region
  inline std::shared_ptr<StencilCall> clone() const;

  /// @brief Comparison between stencils (omitting location)
  inline bool operator==(const StencilCall& rhs) const;

  /// @brief Comparison between stencils (omitting location)
  /// if the comparison fails, outputs human readable reason why in the string
  inline CompareResult comparison(const StencilCall& rhs) const;
};

//===------------------------------------------------------------------------------------------===//
//     StencilCallDeclStmt
//===------------------------------------------------------------------------------------------===//

/// @brief This represents a declaration of a StencilCall
/// @ingroup ast
template <typename DataTraits>
class StencilCallDeclStmt : public DataTraits::StencilCallDeclStmt, public Stmt<DataTraits> {
  std::shared_ptr<StencilCall> stencilCall_;

public:
  USING_STMT_BASE_NAMES
  /// @name Constructor & Destructor
  /// @{
  StencilCallDeclStmt(const std::shared_ptr<StencilCall>& stencilCall,
                      SourceLocation loc = SourceLocation());
  StencilCallDeclStmt(const StencilCallDeclStmt<DataTraits>& stmt);
  StencilCallDeclStmt<DataTraits>& operator=(StencilCallDeclStmt<DataTraits> stmt);
  virtual ~StencilCallDeclStmt();
  /// @}

  const std::shared_ptr<StencilCall>& getStencilCall() const { return stencilCall_; }

  virtual bool isStencilDesc() const override { return true; }
  virtual std::shared_ptr<Stmt<DataTraits>> clone() const override;
  virtual bool equals(const Stmt<DataTraits>* other) const override;
  static bool classof(const Stmt<DataTraits>* stmt) {
    return stmt->getKind() == SK_StencilCallDeclStmt;
  }
  ACCEPTVISITOR(Stmt<DataTraits>, StencilCallDeclStmt<DataTraits>)
};

//===------------------------------------------------------------------------------------------===//
//     BoundaryConditionDeclStmt
//===------------------------------------------------------------------------------------------===//

/// @brief This represents a declaration of a boundary condition
/// @ingroup ast
template <typename DataTraits>
class BoundaryConditionDeclStmt : public DataTraits::BoundaryConditionDeclStmt,
                                  public Stmt<DataTraits> {
  std::string functor_;
  std::vector<std::string> fields_;

public:
  USING_STMT_BASE_NAMES
  /// @name Constructor & Destructor
  /// @{
  BoundaryConditionDeclStmt(const std::string& callee, SourceLocation loc = SourceLocation());
  BoundaryConditionDeclStmt(const BoundaryConditionDeclStmt<DataTraits>& stmt);
  BoundaryConditionDeclStmt<DataTraits>& operator=(BoundaryConditionDeclStmt<DataTraits> stmt);
  virtual ~BoundaryConditionDeclStmt();
  /// @}

  const std::string& getFunctor() const { return functor_; }

  std::vector<std::string>& getFields() { return fields_; }
  const std::vector<std::string>& getFields() const { return fields_; }

  virtual bool isStencilDesc() const override { return true; }
  virtual std::shared_ptr<Stmt<DataTraits>> clone() const override;
  virtual bool equals(const Stmt<DataTraits>* other) const override;
  static bool classof(const Stmt<DataTraits>* stmt) {
    return stmt->getKind() == SK_BoundaryConditionDeclStmt;
  }
  ACCEPTVISITOR(Stmt<DataTraits>, BoundaryConditionDeclStmt<DataTraits>)
};

//===------------------------------------------------------------------------------------------===//
//     IfStmt
//===------------------------------------------------------------------------------------------===//

/// @brief This represents an if/then/else block
/// @ingroup ast
template <typename DataTraits>
class IfStmt : public DataTraits::IfStmt, public Stmt<DataTraits> {
  enum OperandKind { OK_Cond, OK_Then, OK_Else, OK_End };
  std::shared_ptr<Stmt<DataTraits>> subStmts_[OK_End];

public:
  USING_STMT_BASE_NAMES
  /// @name Constructor & Destructor
  /// @{
  IfStmt(const std::shared_ptr<Stmt<DataTraits>>& condExpr,
         const std::shared_ptr<Stmt<DataTraits>>& thenStmt,
         const std::shared_ptr<Stmt<DataTraits>>& elseStmt = nullptr,
         SourceLocation loc = SourceLocation());
  IfStmt(const IfStmt<DataTraits>& stmt);
  IfStmt<DataTraits>& operator=(IfStmt<DataTraits> stmt);
  virtual ~IfStmt();
  /// @}

  const std::shared_ptr<Expr<DataTraits>>& getCondExpr() const {
    return dyn_cast<ExprStmt<DataTraits>>(subStmts_[OK_Cond].get())->getExpr();
  }
  std::shared_ptr<Expr<DataTraits>>& getCondExpr() {
    return dyn_cast<ExprStmt<DataTraits>>(subStmts_[OK_Cond].get())->getExpr();
  }

  const std::shared_ptr<Stmt<DataTraits>>& getCondStmt() const { return subStmts_[OK_Cond]; }
  std::shared_ptr<Stmt<DataTraits>>& getCondStmt() { return subStmts_[OK_Cond]; }

  const std::shared_ptr<Stmt<DataTraits>>& getThenStmt() const { return subStmts_[OK_Then]; }
  std::shared_ptr<Stmt<DataTraits>>& getThenStmt() { return subStmts_[OK_Then]; }
  void setThenStmt(std::shared_ptr<Stmt<DataTraits>>& thenStmt) { subStmts_[OK_Then] = thenStmt; }

  const std::shared_ptr<Stmt<DataTraits>>& getElseStmt() const { return subStmts_[OK_Else]; }
  std::shared_ptr<Stmt<DataTraits>>& getElseStmt() { return subStmts_[OK_Else]; }
  bool hasElse() const { return getElseStmt() != nullptr; }
  void setElseStmt(std::shared_ptr<Stmt<DataTraits>>& elseStmt) { subStmts_[OK_Else] = elseStmt; }

  virtual std::shared_ptr<Stmt<DataTraits>> clone() const override;
  virtual bool equals(const Stmt<DataTraits>* other) const override;
  static bool classof(const Stmt<DataTraits>* stmt) { return stmt->getKind() == SK_IfStmt; }
  virtual StmtRangeType getChildren() override {
    return hasElse() ? StmtRangeType(subStmts_) : StmtRangeType(&subStmts_[0], OK_End - 1);
  }
  virtual void replaceChildren(const std::shared_ptr<Stmt<DataTraits>>& oldStmt,
                               const std::shared_ptr<Stmt<DataTraits>>& newStmt) override;
  ACCEPTVISITOR(Stmt<DataTraits>, IfStmt<DataTraits>)
};
} // namespace ast
} // namespace dawn

#include "dawn/AST/ASTStmt.tcc"

#endif
