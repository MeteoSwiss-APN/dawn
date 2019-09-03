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

#include "dawn/AST/ASTVisitor.h"

#define ACCEPTVISITOR(subtype, type)                                                               \
  virtual inline void accept(ASTVisitor<DataTraits>& visitor) override {                           \
    visitor.visit(std::static_pointer_cast<type>(this->shared_from_this()));                       \
  }                                                                                                \
  virtual inline void accept(ASTVisitorNonConst<DataTraits>& visitor) override {                   \
    visitor.visit(std::static_pointer_cast<type>(this->shared_from_this()));                       \
  }                                                                                                \
  virtual inline std::shared_ptr<subtype> acceptAndReplace(                                        \
      ASTVisitorPostOrder<DataTraits>& visitor) override {                                         \
    return visitor.visitAndReplace(std::static_pointer_cast<type>(this->shared_from_this()));      \
  }
