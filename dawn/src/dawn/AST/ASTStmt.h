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

#pragma once

#include "dawn/AST/ASTVisitorHelpers.h"
#include "dawn/AST/LocationType.h"
#include "dawn/Support/ArrayRef.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/ComparisonHelpers.h"
#include "dawn/Support/SourceLocation.h"
#include "dawn/Support/Type.h"
#include "dawn/Support/UIDGenerator.h"
#include <memory>
#include <sstream>
#include <vector>

namespace dawn {

namespace sir {
struct VerticalRegion;
} // namespace sir

namespace ast {
class ASTVisitor;
class Expr;

struct StmtData {
  enum DataType { SIR_DATA_TYPE, IIR_DATA_TYPE };
  virtual ~StmtData() {}
  virtual DataType getDataType() const = 0;
  virtual std::unique_ptr<StmtData> clone() const = 0;
  virtual bool equals(StmtData const* other) const = 0;
};

/// @brief Abstract base class of all statements
/// @ingroup ast
class Stmt : public std::enable_shared_from_this<Stmt> {
public:
  /// @brief Discriminator for RTTI (dyn_cast<> et al.)
  enum class Kind {
    BlockStmt,
    ExprStmt,
    ReturnStmt,
    VarDeclStmt,
    StencilCallDeclStmt,
    VerticalRegionDeclStmt,
    BoundaryConditionDeclStmt,
    IfStmt,
    LoopStmt,
  };

  using StmtRangeType = MutableArrayRef<std::shared_ptr<Stmt>>;

  /// @name Constructor & Destructor
  /// @{
  Stmt(std::unique_ptr<StmtData> data, Kind kind, SourceLocation loc = SourceLocation())
      : kind_(kind), loc_(loc), statementID_(UIDGenerator::getInstance()->get()),
        data_(std::move(data)) {}
  Stmt(const Stmt& stmt)
      : std::enable_shared_from_this<Stmt>(stmt), kind_(stmt.getKind()),
        loc_(stmt.getSourceLocation()), statementID_(UIDGenerator::getInstance()->get()),
        data_(stmt.data_->clone()) {}
  virtual ~Stmt() {}
  /// @}

  /// @brief Hook for Visitors
  virtual void accept(ASTVisitor& visitor) = 0;
  virtual void accept(ASTVisitorNonConst& visitor) = 0;
  virtual std::shared_ptr<Stmt> acceptAndReplace(ASTVisitorPostOrder& visitor) = 0;

  /// @brief Clone the current statement
  virtual std::shared_ptr<Stmt> clone() const = 0;

  /// @brief Get kind of Stmt (used by RTTI dyn_cast<> et al.)
  Kind getKind() const { return kind_; }

  /// @brief Get original source location
  const SourceLocation& getSourceLocation() const { return loc_; }
  SourceLocation& getSourceLocation() { return loc_; }

  template <typename Sub, typename Base>
  using enable_if_subtype_t = typename std::enable_if<std::is_base_of<Base, Sub>::value>::type;

  /// @brief Dynamically determine the data type
  StmtData::DataType getDataType() const { return data_->getDataType(); }

  /// @brief Get data object, must provide the type of the data object (must be subtype of StmtData)
  template <typename DataType, typename = enable_if_subtype_t<DataType, StmtData>>
  DataType& getData() {
    DAWN_ASSERT_MSG(DataType::ThisDataType == data_->getDataType(),
                    "Trying to get wrong data type");
    return dynamic_cast<DataType&>(*data_.get());
  }
  template <typename DataType, typename = enable_if_subtype_t<DataType, StmtData>>
  const DataType& getData() const {
    DAWN_ASSERT_MSG(DataType::ThisDataType == data_->getDataType(),
                    "Trying to get wrong data type");
    return dynamic_cast<DataType&>(*data_.get());
  }

  /// @brief Iterate children (if any)
  virtual StmtRangeType getChildren() { return StmtRangeType(); }

  virtual void replaceChildren(std::shared_ptr<Stmt> const& oldStmt,
                               std::shared_ptr<Stmt> const& newStmt) {}

  /// @brief Compare for equality
  virtual bool equals(const Stmt* other, bool compareData = true) const {
    return kind_ == other->kind_ &&
           (compareData ? checkSameDataType(*other) && data_->equals(other->data_.get()) : true);
  }

  /// @brief Is the statement used for stencil description and has no real analogon in C++
  /// (e.g a VerticalRegion or StencilCall)?
  virtual bool isStencilDesc() const { return false; }

  /// @brief Check if statements have the same runtime data type
  bool checkSameDataType(const Stmt& other) const {
    return data_->getDataType() == other.data_->getDataType();
  }

  /// @name Operators
  /// @{
  bool operator==(const Stmt& other) const { return other.equals(this); }
  bool operator!=(const Stmt& other) const { return !(*this == other); }
  /// @}

  /// @brief get the statementID for mapping
  int getID() const { return statementID_; }

  void setID(int id) { statementID_ = id; }

protected:
  // copy assignment
  void assign(const Stmt& other) {
    DAWN_ASSERT_MSG((checkSameDataType(other)), "Trying to assign Stmt with different data type");
    kind_ = other.kind_;
    loc_ = other.loc_;
    data_ = other.data_->clone();
  }

  Kind kind_;
  SourceLocation loc_;

  int statementID_;

private:
  std::unique_ptr<StmtData> data_;
};

//===------------------------------------------------------------------------------------------===//
//     BlockStmt
//===------------------------------------------------------------------------------------------===//

/// @brief Block of statements
/// @ingroup ast
class BlockStmt : public Stmt {
  std::vector<std::shared_ptr<Stmt>> statements_;

public:
  using StatementList = std::vector<std::shared_ptr<Stmt>>;
  using StmtConstIterator = StatementList::const_iterator;
  using StmtIterator = StatementList::iterator;

  /// @name Constructor & Destructor
  /// @{
  BlockStmt(std::unique_ptr<StmtData> data, SourceLocation loc = SourceLocation());
  BlockStmt(std::unique_ptr<StmtData> data, const std::vector<std::shared_ptr<Stmt>>& statements,
            SourceLocation loc = SourceLocation());
  BlockStmt(const BlockStmt& stmt);
  BlockStmt& operator=(BlockStmt const& stmt);
  virtual ~BlockStmt();
  /// @}

  /// @brief inserts stmts from (iterators) `first` to `last` into the block at `position`
  template <typename InputIterator>
  StmtIterator insert(StmtConstIterator position, InputIterator first, InputIterator last) {
#if DAWN_USING_ASSERTS
    std::for_each(first, last, [&](const std::shared_ptr<Stmt>& stmt) {
      DAWN_ASSERT(stmt);
      DAWN_ASSERT_MSG((checkSameDataType(*stmt)),
                      "Trying to insert child Stmt with different data type");
    });
#endif
    return statements_.insert(position, first, last);
  }

  /// @brief inserts stmts in `range` at the end of the block
  template <class Range>
  void insert_back(Range&& range) {
    insert_back(std::begin(range), std::end(range));
  }

  /// @brief inserts stmts from (iterators) `begin` to `end` at the end of the block
  template <class InputIterator>
  void insert_back(InputIterator begin, InputIterator end) {
    insert(statements_.end(), begin, end);
  }

  /// @brief inserts `stmt` at the end of the block
  void push_back(std::shared_ptr<Stmt>&& stmt);

  /// @brief substitutes stmt at `position` with `replacement`
  void substitute(StmtConstIterator position, std::shared_ptr<Stmt>&& replacement);

  /// @brief removes stmt at `position` from the block
  StmtIterator erase(StmtConstIterator position) { return statements_.erase(position); }

  /// @brief removes all stmts from the block
  void clear() { statements_.clear(); }

  /// @brief returns a const reference to the container of the statements in the block
  const std::vector<std::shared_ptr<Stmt>>& getStatements() const { return statements_; }

  virtual std::shared_ptr<Stmt> clone() const override;
  virtual bool equals(const Stmt* other, bool compareData = true) const override;
  static bool classof(const Stmt* stmt) { return stmt->getKind() == Kind::BlockStmt; }
  virtual StmtRangeType getChildren() override { return StmtRangeType(statements_); }
  virtual void replaceChildren(const std::shared_ptr<Stmt>& oldStmt,
                               const std::shared_ptr<Stmt>& newStmt) override;

  /// @brief whether it's an empty block or not
  bool isEmpty() const { return statements_.empty(); }

  ACCEPTVISITOR(Stmt, BlockStmt)
};

//===------------------------------------------------------------------------------------------===//
//     ExprStmt
//===------------------------------------------------------------------------------------------===//

/// @brief Block of statements
/// @ingroup ast
class ExprStmt : public Stmt {
  std::shared_ptr<Expr> expr_;

public:
  /// @name Constructor & Destructor
  /// @{
  ExprStmt(std::unique_ptr<StmtData> data, const std::shared_ptr<Expr>& expr,
           SourceLocation loc = SourceLocation());
  ExprStmt(const ExprStmt& stmt);
  ExprStmt& operator=(ExprStmt stmt);
  virtual ~ExprStmt();
  /// @}

  void setExpr(const std::shared_ptr<Expr>& expr) { expr_ = expr; }
  const std::shared_ptr<Expr>& getExpr() const { return expr_; }
  std::shared_ptr<Expr>& getExpr() { return expr_; }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
  virtual void replaceChildren(const std::shared_ptr<Expr>& oldExpr,
                               const std::shared_ptr<Expr>& newExpr);
#pragma GCC diagnostic pop

  virtual std::shared_ptr<Stmt> clone() const override;
  virtual bool equals(const Stmt* other, bool compareData = true) const override;
  static bool classof(const Stmt* stmt) { return stmt->getKind() == Kind::ExprStmt; }
  ACCEPTVISITOR(Stmt, ExprStmt)
};

//===------------------------------------------------------------------------------------------===//
//     ReturnStmt
//===------------------------------------------------------------------------------------------===//

/// @brief This represents a return of an expression
/// @ingroup ast
class ReturnStmt : public Stmt {
  std::shared_ptr<Expr> expr_;

public:
  /// @name Constructor & Destructor
  /// @{
  ReturnStmt(std::unique_ptr<StmtData> data, const std::shared_ptr<Expr>& expr,
             SourceLocation loc = SourceLocation());
  ReturnStmt(const ReturnStmt& stmt);
  ReturnStmt& operator=(ReturnStmt stmt);
  virtual ~ReturnStmt();
  /// @}

  void setExpr(const std::shared_ptr<Expr>& expr) { expr_ = expr; }
  const std::shared_ptr<Expr>& getExpr() const { return expr_; }
  std::shared_ptr<Expr>& getExpr() { return expr_; }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
  virtual void replaceChildren(const std::shared_ptr<Expr>& oldExpr,
                               const std::shared_ptr<Expr>& newExpr);
#pragma GCC diagnostic pop

  virtual std::shared_ptr<Stmt> clone() const override;
  virtual bool equals(const Stmt* other, bool compareData = true) const override;
  static bool classof(const Stmt* stmt) { return stmt->getKind() == Kind::ReturnStmt; }
  ACCEPTVISITOR(Stmt, ReturnStmt)
};

//===------------------------------------------------------------------------------------------===//
//     VarDeclStmt
//===------------------------------------------------------------------------------------------===//

/// @brief This represents a declaration of a local variable or C-array
/// @ingroup ast
class VarDeclStmt : public Stmt {
public:
  using InitList = std::vector<std::shared_ptr<Expr>>;

  /// @name Constructor & Destructor
  /// @{
  VarDeclStmt(std::unique_ptr<StmtData> data, const Type& type, const std::string& name,
              int dimension, const std::string& op, InitList initList,
              SourceLocation loc = SourceLocation());
  VarDeclStmt(const VarDeclStmt& stmt);
  VarDeclStmt& operator=(VarDeclStmt stmt);
  virtual ~VarDeclStmt();
  /// @}

  const Type& getType() const { return type_; }
  Type& getType() { return type_; }

  const std::string& getName() const { return name_; }
  std::string& getName() { return name_; }

  const std::string& getOp() const { return op_; }
  int getDimension() const { return dimension_; }

  bool isArray() const { return (dimension_ > 0); }
  bool hasInit() const { return (!initList_.empty()); }
  const InitList& getInitList() const { return initList_; }
  InitList& getInitList() { return initList_; }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
  virtual void replaceChildren(const std::shared_ptr<Expr>& oldExpr,
                               const std::shared_ptr<Expr>& newExpr);
#pragma GCC diagnostic pop

  virtual std::shared_ptr<Stmt> clone() const override;
  virtual bool equals(const Stmt* other, bool compareData = true) const override;
  static bool classof(const Stmt* stmt) { return stmt->getKind() == Kind::VarDeclStmt; }
  ACCEPTVISITOR(Stmt, VarDeclStmt)

private:
  Type type_;
  std::string name_;

  // Dimension of the array or 0 for variables
  int dimension_;
  std::string op_;

  // List of expression used for initializaion or just 1 element for variables
  InitList initList_;
};

//===------------------------------------------------------------------------------------------===//
//     VerticalRegionDeclStmt
//===------------------------------------------------------------------------------------------===//

/// @brief This represents a declaration of a sir::VerticalRegion
/// @ingroup ast
class VerticalRegionDeclStmt : public Stmt {
  std::shared_ptr<sir::VerticalRegion> verticalRegion_;

public:
  /// @name Constructor & Destructor
  /// @{
  VerticalRegionDeclStmt(std::unique_ptr<StmtData> data,
                         const std::shared_ptr<sir::VerticalRegion>& verticalRegion,
                         SourceLocation loc = SourceLocation());
  VerticalRegionDeclStmt(const VerticalRegionDeclStmt& stmt);
  VerticalRegionDeclStmt& operator=(VerticalRegionDeclStmt stmt);
  virtual ~VerticalRegionDeclStmt();
  /// @}

  const std::shared_ptr<sir::VerticalRegion>& getVerticalRegion() const { return verticalRegion_; }

  virtual bool isStencilDesc() const override { return true; }
  virtual std::shared_ptr<Stmt> clone() const override;
  virtual bool equals(const Stmt* other, bool compareData = true) const override;
  static bool classof(const Stmt* stmt) { return stmt->getKind() == Kind::VerticalRegionDeclStmt; }
  ACCEPTVISITOR(Stmt, VerticalRegionDeclStmt)
};

/// @brief Call to another stencil
/// @ingroup ast
struct StencilCall {

  SourceLocation Loc;            ///< Source location of the call
  std::string Callee;            ///< Name of the callee stencil
  std::vector<std::string> Args; ///< List of fields used as arguments

  StencilCall(std::string callee, SourceLocation loc = SourceLocation())
      : Loc(loc), Callee(callee) {}

  /// @brief Clone the stencil call
  std::shared_ptr<StencilCall> clone() const;

  /// @brief Comparison between stencils (omitting location)
  bool operator==(const StencilCall& rhs) const;

  /// @brief Comparison between stencils (omitting location)
  /// if the comparison fails, outputs human readable reason why in the string
  CompareResult comparison(const StencilCall& rhs) const;
};

//===------------------------------------------------------------------------------------------===//
//     StencilCallDeclStmt
//===------------------------------------------------------------------------------------------===//

/// @brief This represents a declaration of a StencilCall
/// @ingroup ast
class StencilCallDeclStmt : public Stmt {
  std::shared_ptr<StencilCall> stencilCall_;

public:
  /// @name Constructor & Destructor
  /// @{
  StencilCallDeclStmt(std::unique_ptr<StmtData> data,
                      const std::shared_ptr<StencilCall>& stencilCall,
                      SourceLocation loc = SourceLocation());
  StencilCallDeclStmt(const StencilCallDeclStmt& stmt);
  StencilCallDeclStmt& operator=(StencilCallDeclStmt stmt);
  virtual ~StencilCallDeclStmt();
  /// @}

  const std::shared_ptr<StencilCall>& getStencilCall() const { return stencilCall_; }

  virtual bool isStencilDesc() const override { return true; }
  virtual std::shared_ptr<Stmt> clone() const override;
  virtual bool equals(const Stmt* other, bool compareData = true) const override;
  static bool classof(const Stmt* stmt) { return stmt->getKind() == Kind::StencilCallDeclStmt; }
  ACCEPTVISITOR(Stmt, StencilCallDeclStmt)
};

//===------------------------------------------------------------------------------------------===//
//     BoundaryConditionDeclStmt
//===------------------------------------------------------------------------------------------===//

/// @brief This represents a declaration of a boundary condition
/// @ingroup ast
class BoundaryConditionDeclStmt : public Stmt {
  std::string functor_;
  std::vector<std::string> fields_;

public:
  /// @name Constructor & Destructor
  /// @{
  BoundaryConditionDeclStmt(std::unique_ptr<StmtData> data, const std::string& callee,
                            SourceLocation loc = SourceLocation());
  BoundaryConditionDeclStmt(const BoundaryConditionDeclStmt& stmt);
  BoundaryConditionDeclStmt& operator=(BoundaryConditionDeclStmt stmt);
  virtual ~BoundaryConditionDeclStmt();
  /// @}

  const std::string& getFunctor() const { return functor_; }

  std::vector<std::string>& getFields() { return fields_; }
  const std::vector<std::string>& getFields() const { return fields_; }

  virtual bool isStencilDesc() const override { return true; }
  virtual std::shared_ptr<Stmt> clone() const override;
  virtual bool equals(const Stmt* other, bool compareData = true) const override;
  static bool classof(const Stmt* stmt) {
    return stmt->getKind() == Kind::BoundaryConditionDeclStmt;
  }
  ACCEPTVISITOR(Stmt, BoundaryConditionDeclStmt)
};

//===------------------------------------------------------------------------------------------===//
//     IfStmt
//===------------------------------------------------------------------------------------------===//

/// @brief This represents an if/then/else block
/// @ingroup ast
class IfStmt : public Stmt {
  enum OperandKind { Cond = 0, Then, Else, End };
  std::shared_ptr<Stmt> subStmts_[End];

public:
  /// @name Constructor & Destructor
  /// @{
  IfStmt(std::unique_ptr<StmtData> data, const std::shared_ptr<Stmt>& condExpr,
         const std::shared_ptr<Stmt>& thenStmt, const std::shared_ptr<Stmt>& elseStmt = nullptr,
         SourceLocation loc = SourceLocation());
  IfStmt(const IfStmt& stmt);
  IfStmt& operator=(IfStmt stmt);
  virtual ~IfStmt();
  /// @}

  // TODO refactor_AST: this non-const getters are sources of problems: no runtime checks on data
  // type when user changes the substatements! (should have only setters and const getters)

  const std::shared_ptr<Expr>& getCondExpr() const {
    return dyn_cast<ExprStmt>(subStmts_[Cond].get())->getExpr();
  }
  std::shared_ptr<Expr>& getCondExpr() {
    return dyn_cast<ExprStmt>(subStmts_[Cond].get())->getExpr();
  }

  const std::shared_ptr<Stmt>& getCondStmt() const { return subStmts_[Cond]; }
  std::shared_ptr<Stmt>& getCondStmt() { return subStmts_[Cond]; }

  const std::shared_ptr<Stmt>& getThenStmt() const { return subStmts_[Then]; }
  std::shared_ptr<Stmt>& getThenStmt() { return subStmts_[Then]; }
  void setThenStmt(std::shared_ptr<Stmt>& thenStmt) {
    DAWN_ASSERT_MSG((checkSameDataType(*thenStmt)),
                    "Trying to set substmt with different data type");
    subStmts_[Then] = thenStmt;
  }

  const std::shared_ptr<Stmt>& getElseStmt() const { return subStmts_[Else]; }
  std::shared_ptr<Stmt>& getElseStmt() { return subStmts_[Else]; }
  bool hasElse() const { return getElseStmt() != nullptr; }
  void setElseStmt(std::shared_ptr<Stmt>& elseStmt) {
    DAWN_ASSERT_MSG((checkSameDataType(*elseStmt)),
                    "Trying to set substmt with different data type");
    subStmts_[Else] = elseStmt;
  }

  virtual std::shared_ptr<Stmt> clone() const override;
  virtual bool equals(const Stmt* other, bool compareData = true) const override;
  static bool classof(const Stmt* stmt) { return stmt->getKind() == Kind::IfStmt; }
  virtual StmtRangeType getChildren() override {
    return hasElse() ? StmtRangeType(subStmts_) : StmtRangeType(&subStmts_[0], End - 1);
  }
  virtual void replaceChildren(const std::shared_ptr<Stmt>& oldStmt,
                               const std::shared_ptr<Stmt>& newStmt) override;
  ACCEPTVISITOR(Stmt, IfStmt)
};

//===------------------------------------------------------------------------------------------===//
//     Loop
//===------------------------------------------------------------------------------------------===//

class IterationDescr {
public:
  virtual ~IterationDescr() = 0;
  virtual std::unique_ptr<IterationDescr> clone() const = 0;
  virtual std::string toString() const = 0;
  virtual bool equals(const IterationDescr*) const = 0;
};

class ChainIterationDescr : public IterationDescr {
  ast::NeighborChain chain_;

public:
  ChainIterationDescr(ast::NeighborChain&& chain);
  ast::NeighborChain getChain() const;
  std::unique_ptr<IterationDescr> clone() const override;
  std::string toString() const override;
  bool equals(const IterationDescr* otherPtr) const override;
};

class LoopStmt : public Stmt {
  std::shared_ptr<BlockStmt> blockStmt_;
  std::unique_ptr<IterationDescr> iterationDescr_;

public:
  LoopStmt(std::unique_ptr<StmtData> data, ast::NeighborChain&& chain,
           std::shared_ptr<BlockStmt> stmt, SourceLocation loc = SourceLocation());
  LoopStmt(const LoopStmt& stmt);
  LoopStmt& operator=(LoopStmt const& stmt);
  virtual ~LoopStmt();

  const std::shared_ptr<BlockStmt>& getBlockStmt() const;
  std::shared_ptr<BlockStmt>& getBlockStmt();

  std::shared_ptr<Stmt> clone() const override;
  bool equals(const Stmt* other, bool compareData = true) const override;
  static bool classof(const Stmt* stmt) { return stmt->getKind() == Kind::LoopStmt; }
  virtual StmtRangeType getChildren() override;
  virtual void replaceChildren(const std::shared_ptr<Stmt>& oldStmt,
                               const std::shared_ptr<Stmt>& newStmt) override;

  const IterationDescr& getIterationDescr() const;
  const IterationDescr* getIterationDescrPtr() const;

  ACCEPTVISITOR(Stmt, LoopStmt)
};

} // namespace ast
} // namespace dawn
