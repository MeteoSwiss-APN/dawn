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

#include "dawn/AST/ASTStmt.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Casting.h"

namespace dawn {
namespace ast {
template <typename DataTraits>
AST<DataTraits>::AST() : root_(std::make_shared<BlockStmt<DataTraits>>()) {}

template <typename DataTraits>
AST<DataTraits>::AST(const std::shared_ptr<BlockStmt<DataTraits>>& root) : root_(root) {
  DAWN_ASSERT(root_ != nullptr);
}

template <typename DataTraits>
AST<DataTraits>::AST(std::shared_ptr<BlockStmt<DataTraits>>&& root) : root_(std::move(root)) {
  DAWN_ASSERT(root_ != nullptr);
}

template <typename DataTraits>
AST<DataTraits>::~AST() {}

template <typename DataTraits>
AST<DataTraits>::AST(const AST<DataTraits>& ast) {
  root_ = std::static_pointer_cast<BlockStmt<DataTraits>>(ast.getRoot()->clone());
}

template <typename DataTraits>
AST<DataTraits>& AST<DataTraits>::operator=(const AST<DataTraits>& ast) {
  root_ = std::static_pointer_cast<BlockStmt<DataTraits>>(ast.getRoot()->clone());
  return *this;
}

template <typename DataTraits>
AST<DataTraits>::AST(AST<DataTraits>&& ast) {
  root_ = std::move(ast.getRoot());
}

template <typename DataTraits>
AST<DataTraits>& AST<DataTraits>::operator=(AST<DataTraits>&& ast) {
  root_ = std::move(ast.getRoot());
  return *this;
}

template <typename DataTraits>
void AST<DataTraits>::accept(ASTVisitor<DataTraits>& visitor) const {
  root_->accept(visitor);
}

template <typename DataTraits>
void AST<DataTraits>::accept(ASTVisitorNonConst<DataTraits>& visitor) const {
  root_->accept(visitor);
}

template <typename DataTraits>
std::shared_ptr<AST<DataTraits>>
AST<DataTraits>::acceptAndReplace(ASTVisitorPostOrder<DataTraits>& visitor) const {
  return std::make_shared<AST<DataTraits>>(
      std::static_pointer_cast<BlockStmt<DataTraits>>(visitor.visitAndReplace(root_)));
}

template <typename DataTraits>
std::shared_ptr<BlockStmt<DataTraits>>& AST<DataTraits>::getRoot() {
  return root_;
}

template <typename DataTraits>
const std::shared_ptr<BlockStmt<DataTraits>>& AST<DataTraits>::getRoot() const {
  return root_;
}

template <typename DataTraits>
void AST<DataTraits>::setRoot(const std::shared_ptr<BlockStmt<DataTraits>>& root) {
  root_ = root;
}

template <typename DataTraits>
void AST<DataTraits>::setRoot(std::shared_ptr<BlockStmt<DataTraits>>&& root) {
  root_ = std::move(root);
}

template <typename DataTraits>
std::shared_ptr<AST<DataTraits>> AST<DataTraits>::clone() const {
  return std::make_shared<AST<DataTraits>>(
      std::static_pointer_cast<BlockStmt<DataTraits>>(root_->clone()));
}

template <typename DataTraits>
void AST<DataTraits>::clear() {
  root_.reset();
}
} // namespace ast
} // namespace dawn
