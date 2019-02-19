//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _       _
//                        | |     | |
//                    __ _| |_ ___| | __ _ _ __   __ _
//                   / _` | __/ __| |/ _` | '_ \ / _` |
//                  | (_| | || (__| | (_| | | | | (_| |
//                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
//                    __/ |                       __/ |
//                   |___/                       |___/
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "clang/AST/ASTFwd.h"
#include "clang/AST/ExprCXX.h"

#include "gtclang/Support/ASTUtils.h"

namespace gtclang {

std::string getClassNameFromConstructExpr(clang::CXXConstructExpr* expr) {
  clang::CXXRecordDecl* recDecl = expr->getConstructor()->getParent();
  return recDecl->getNameAsString();
}
}