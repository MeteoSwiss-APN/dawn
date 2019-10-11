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

namespace gtclang {
namespace clang_compat {

#if CLANG_VERSION_MAJOR < 9
template <typename StmtT>
typename std::enable_if<std::is_base_of<clang::Stmt, typename std::decay<StmtT>::type>::value,
                        StmtT*>::type
skipAllImplicitNodes(StmtT* e) {
  while(e != e->IgnoreImplicit())
    e = e->IgnoreImplicit();
  return e;
}
#else
template <typename StmtT>
typename std::enable_if<std::is_base_of<clang::Stmt, typename std::decay<StmtT>::type>::value,
                        StmtT*>::type
skipAllImplicitNodes(StmtT* s) {
  if(auto* e = llvm::dyn_cast_or_null<clang::Expr>(s)) {
    while(e != e->IgnoreImplicit())
      e = e->IgnoreImplicit();
    return e;
  }
  return s;
}
#endif
} // namespace clang_compat
} // namespace gtclang

#endif // GTCLANG_SUPPORT_CLANGCOMPAT_IMPLICIT_NODES_H
