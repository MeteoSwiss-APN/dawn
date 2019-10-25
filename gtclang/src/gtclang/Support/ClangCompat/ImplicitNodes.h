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

#ifndef GTCLANG_SUPPORT_CLANGCOMPAT_IMPLICIT_NODES_H
#define GTCLANG_SUPPORT_CLANGCOMPAT_IMPLICIT_NODES_H

#include "clang/AST/Expr.h"
#include "clang/Basic/Version.h"

// TODO skipAllImplicitNodes should be refactored to be only called for Exprs (not Stmts)
// to reflect the change in clang 9

namespace gtclang::clang_compat {

#if CLANG_VERSION_MAJOR < 9
template <typename StmtT>
StmtT* skipAllImplicitNodes(StmtT* e) {
  static_assert(std::is_base_of_v<clang::Stmt, std::decay_t<StmtT>>);
  while(e != e->IgnoreImplicit())
    e = e->IgnoreImplicit();
  return e;
}
#else
template <typename StmtT>
StmtT* skipAllImplicitNodes(StmtT* s) {
  static_assert(std::is_base_of_v<clang::Stmt, std::decay_t<StmtT>>);
  if(auto* e = llvm::dyn_cast_or_null<clang::Expr>(s)) {
    while(e != e->IgnoreImplicit())
      e = e->IgnoreImplicit();
    return e;
  }
  return s;
}
#endif
} // namespace gtclang::clang_compat

#endif // GTCLANG_SUPPORT_CLANGCOMPAT_IMPLICIT_NODES_H
