//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#ifndef GSL_SIR_AST_H
#define GSL_SIR_AST_H

#include "gsl/SIR/ASTExpr.h"
#include "gsl/SIR/ASTStmt.h"
#include "gsl/SIR/ASTStringifier.h"
#include <memory>

namespace gsl {

class ASTVisitor;

/// @brief Abstract syntax tree of the SIR
/// @ingroup sir
class AST {
  std::shared_ptr<BlockStmt> root_;

public:
  /// @brief Construct with empty root
  AST();

  /// @brief Construct with existing root
  ///
  /// Note that root has to be of dynamic type `BlockStmt`.
  explicit AST(const std::shared_ptr<BlockStmt>& root);
  explicit AST(std::shared_ptr<BlockStmt>&& root);

  /// @name Copy/Move construct/assign
  /// @{
  AST(const AST&);
  AST& operator=(const AST&);
  AST(AST&&);
  AST& operator=(AST&&);
  /// @}

  /// @brief Deallocate the AST
  ~AST();

  /// @brief Apply the visitor to all nodes of the AST
  void accept(ASTVisitor& visitor) const;

  /// @brief Get root node
  const std::shared_ptr<BlockStmt>& getRoot() const;
  std::shared_ptr<BlockStmt>& getRoot();

  /// @brief Set root (this will call `clear()` first)
  void setRoot(const std::shared_ptr<BlockStmt>& root);
  void setRoot(std::shared_ptr<BlockStmt>&& root);

  /// @brief Clone the AST
  std::shared_ptr<AST> clone() const;

  /// @brief Clear the AST
  void clear();
};

} // namespace gsl

#endif
