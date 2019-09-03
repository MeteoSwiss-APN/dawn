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

#ifndef DAWN_AST_AST_H
#define DAWN_AST_AST_H

#include "dawn/AST/ASTExpr.h"
#include "dawn/AST/ASTStmt.h"
#include "dawn/AST/ASTStringifier.h"
#include <memory>

namespace dawn {
namespace ast {
template <typename DataTraits>
class ASTVisitor;

/// @brief Abstract syntax tree of the AST
/// @ingroup AST
template <typename DataTraits>
class AST {
  std::shared_ptr<BlockStmt<DataTraits>> root_;

public:
  /// @brief Construct with empty root
  AST();

  /// @brief Construct with existing root
  ///
  /// Note that root has to be of dynamic type `BlockStmt`.
  explicit AST(const std::shared_ptr<BlockStmt<DataTraits>>& root);
  explicit AST(std::shared_ptr<BlockStmt<DataTraits>>&& root);

  /// @name Copy/Move construct/assign
  /// @{
  AST(const AST<DataTraits>&);
  AST& operator=(const AST<DataTraits>&);
  AST(AST<DataTraits>&&);
  AST& operator=(AST<DataTraits>&&);
  /// @}

  /// @brief Deallocate the AST
  ~AST();

  /// @brief Apply the visitor to all nodes of the AST
  void accept(ASTVisitor<DataTraits>& visitor) const;
  void accept(ASTVisitorNonConst<DataTraits>& visitor) const;
  std::shared_ptr<AST<DataTraits>> acceptAndReplace(ASTVisitorPostOrder<DataTraits>& visitor) const;

  /// @brief Get root node
  const std::shared_ptr<BlockStmt<DataTraits>>& getRoot() const;
  std::shared_ptr<BlockStmt<DataTraits>>& getRoot();

  /// @brief Set root (this will call `clear()` first)
  void setRoot(const std::shared_ptr<BlockStmt<DataTraits>>& root);
  void setRoot(std::shared_ptr<BlockStmt<DataTraits>>&& root);

  /// @brief Clone the AST
  std::shared_ptr<AST<DataTraits>> clone() const;

  /// @brief Clear the AST
  void clear();
};
} // namespace ast
} // namespace dawn

#include "dawn/AST/AST.tcc"

#endif
