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

#include "gsl/SIR/AST.h"
#include "gsl/SIR/ASTStmt.h"
#include "gsl/Support/Assert.h"
#include "gsl/Support/Casting.h"

namespace gsl {

AST::AST() : root_(std::make_shared<BlockStmt>()) {}

AST::AST(const std::shared_ptr<BlockStmt>& root) : root_(root) { GSL_ASSERT(root_ != nullptr); }

AST::AST(std::shared_ptr<BlockStmt>&& root) : root_(std::move(root)) {
  GSL_ASSERT(root_ != nullptr);
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

std::shared_ptr<BlockStmt>& AST::getRoot() { return root_; }
const std::shared_ptr<BlockStmt>& AST::getRoot() const { return root_; }

void AST::setRoot(const std::shared_ptr<BlockStmt>& root) { root_ = root; }
void AST::setRoot(std::shared_ptr<BlockStmt>&& root) { root_ = std::move(root); }

std::shared_ptr<AST> AST::clone() const {
  return std::make_shared<AST>(std::static_pointer_cast<BlockStmt>(root_->clone()));
}

void AST::clear() { root_.reset(); }

} // namespace gsl
