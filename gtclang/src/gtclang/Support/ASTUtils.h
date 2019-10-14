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

#include <string>
#include <type_traits>

#include "gtclang/Support/ClangCompat/ImplicitNodes.h"
#include "clang/AST/ASTFwd.h"

namespace gtclang {

/// @brief Retrieve a class name for a given cxx construct expr
///
/// @param expr               clang construct expr
///
/// @ingroup support
extern std::string getClassNameFromConstructExpr(clang::CXXConstructExpr* expr);

/// @brief Retrieve a statement and skip all implicit nodes
/// (ExprWithCleanups, MaterializeTemporaryExpr, CXXBindTemporaryExpr, ImplicitCastExpr)
///
/// @param expr               clang stmt
///
/// @ingroup support
template <typename StmtT>
StmtT* skipAllImplicitNodes(StmtT* s) {
  return clang_compat::skipAllImplicitNodes(s);
}
} // namespace gtclang
