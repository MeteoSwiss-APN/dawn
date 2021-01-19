#pragma once

#include "dawn/AST/Interval.h"
#include "dawn/SIR/AST.h"
#include "dawn/SIR/ASTStmt.h"

namespace dawn {

/// @namespace sir
/// @brief This namespace contains a C++ implementation of the SIR specification
/// @ingroup sir
namespace sir {

//===------------------------------------------------------------------------------------------===//
//     StencilDescription
//===------------------------------------------------------------------------------------------===//

/// @brief A vertical region is given by a list of statements (given as an AST) executed on a
/// specific vertical interval in a given loop order
/// @ingroup sir
struct VerticalRegion {
  enum class LoopOrderKind { Forward, Backward };

  SourceLocation Loc;                         ///< Source location of the vertical region
  std::shared_ptr<ast::AST> Ast;              ///< AST of the region
  std::shared_ptr<ast::Interval> VerticalInterval; ///< Interval description of the region
  LoopOrderKind LoopOrder;                    ///< Loop order (usually associated with the k-loop)

  /// If it is not instantiated, iteration over the full domain is assumed.
  std::array<std::optional<ast::Interval>, 2> IterationSpace; /// < Iteration space in the horizontal.

  VerticalRegion(const std::shared_ptr<ast::AST>& ast,
                 const std::shared_ptr<ast::Interval>& verticalInterval, LoopOrderKind loopOrder,
                 SourceLocation loc = SourceLocation())
      : Loc(loc), Ast(ast), VerticalInterval(verticalInterval), LoopOrder(loopOrder) {}
  VerticalRegion(const std::shared_ptr<ast::AST>& ast,
                 const std::shared_ptr<ast::Interval>& verticalInterval, LoopOrderKind loopOrder,
                 std::optional<ast::Interval> iterationSpaceI, std::optional<ast::Interval> iterationSpaceJ,
                 SourceLocation loc = SourceLocation())
      : Loc(loc), Ast(ast), VerticalInterval(verticalInterval), LoopOrder(loopOrder),
        IterationSpace({iterationSpaceI, iterationSpaceJ}) {}

  /// @brief Clone the vertical region
  std::shared_ptr<VerticalRegion> clone() const;

  /// @brief Comparison between stencils (omitting location)
  bool operator==(const VerticalRegion& rhs) const;

  /// @brief Comparison between stencils (omitting location)
  /// if the comparison fails, outputs human readable reason why in the string
  CompareResult comparison(const VerticalRegion& rhs) const;
};

/// @brief Compares two ASTs
std::pair<std::string, bool> compareAst(const std::shared_ptr<ast::AST>& lhs,
                                        const std::shared_ptr<ast::AST>& rhs);
} // namespace sir

namespace ast {
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

} // namespace ast
} // namespace dawn