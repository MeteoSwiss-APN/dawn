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
#include "dawn/AST/ASTVisitor.h"

#define ACCEPTVISITOR(subtype, type)                                                               \
  virtual inline void accept(ASTVisitor& visitor) override {                                       \
    visitor.visit(std::static_pointer_cast<type>(shared_from_this()));                             \
  }                                                                                                \
  virtual inline void accept(ASTVisitorNonConst& visitor) override {                               \
    visitor.visit(std::static_pointer_cast<type>(shared_from_this()));                             \
  }                                                                                                \
  virtual inline std::shared_ptr<subtype> acceptAndReplace(ASTVisitorPostOrder& visitor)           \
      override {                                                                                   \
    return visitor.visitAndReplace(std::static_pointer_cast<type>(shared_from_this()));            \
  }
