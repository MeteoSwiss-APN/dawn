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

#include "dawn/SIR/AST.h"
#include "dawn/SIR/ASTStmt.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Casting.h"

namespace dawn {

AST::AST() : root_(std::make_shared<BlockStmt>()) {}

AST::AST(const std::shared_ptr<BlockStmt>& root) : root_(root) { DAWN_ASSERT(root_ != nullptr); }

AST::AST(std::shared_ptr<BlockStmt>&& root) : root_(std::move(root)) {
  DAWN_ASSERT(root_ != nullptr);
}

AST::~AST() {}

AST::AST(const AST& ast) { root_ = std::static_pointer_cast<BlockStmt>(ast.getRoot()->clone()); }

AST& AST::operator=(const AST& ast) {
  root_ = std::static_pointer_cast<BlockStmt>(ast.getRoot()->clone());
  return *this;
}

AST::AST(AST&& ast) { root_ = std::move(ast.getRoot()); }

AST& AST::operator=(AST&& ast) {
  root_ = std::move(ast.getRoot());
  return *this;
}

void AST::accept(ASTVisitor& visitor) const { root_->accept(visitor); }

void AST::accept(ASTVisitorNonConst& visitor) const { root_->accept(visitor); }

std::shared_ptr<AST> AST::acceptAndReplace(ASTVisitorPostOrder& visitor) const {
  return std::make_shared<AST>(std::static_pointer_cast<BlockStmt>(visitor.visitAndReplace(root_)));
}

std::shared_ptr<BlockStmt>& AST::getRoot() { return root_; }
const std::shared_ptr<BlockStmt>& AST::getRoot() const { return root_; }

void AST::setRoot(const std::shared_ptr<BlockStmt>& root) { root_ = root; }
void AST::setRoot(std::shared_ptr<BlockStmt>&& root) { root_ = std::move(root); }

std::shared_ptr<AST> AST::clone() const {
  return std::make_shared<AST>(std::static_pointer_cast<BlockStmt>(root_->clone()));
}

void AST::clear() { root_.reset(); }

} // namespace dawn
