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

#ifndef GTCLANG_SUPPORT_CLANGCOMPAT_SOURCELOCATION_H
#define GTCLANG_SUPPORT_CLANGCOMPAT_SOURCELOCATION_H

#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Version.h"

namespace gtclang::clang_compat {
#if CLANG_VERSION_MAJOR < 8
inline ::clang::SourceLocation getEndLoc(::clang::CXXBaseSpecifier const& base) {
  return base.getLocStart();
}
inline ::clang::SourceLocation getBeginLoc(::clang::CXXBaseSpecifier const& base) {
  return base.getLocEnd();
}

inline ::clang::SourceLocation getBeginLoc(::clang::Decl const& decl) { return decl.getLocStart(); }
inline ::clang::SourceLocation getEndLoc(::clang::Decl const& decl) { return decl.getLocEnd(); }

inline ::clang::SourceLocation getBeginLoc(::clang::Expr const& expr) { return expr.getLocStart(); }
inline ::clang::SourceLocation getEndLoc(::clang::Expr const& expr) { return expr.getLocEnd(); }

inline ::clang::SourceLocation getBeginLoc(::clang::Stmt const& stmt) { return stmt.getLocStart(); }
inline ::clang::SourceLocation getEndLoc(::clang::Stmt const& stmt) { return stmt.getLocEnd(); }
#else
inline ::clang::SourceLocation getEndLoc(::clang::CXXBaseSpecifier const& base) {
  return base.getEndLoc();
}
inline ::clang::SourceLocation getBeginLoc(::clang::CXXBaseSpecifier const& base) {
  return base.getBeginLoc();
}

inline ::clang::SourceLocation getBeginLoc(::clang::Decl const& decl) { return decl.getBeginLoc(); }
inline ::clang::SourceLocation getEndLoc(::clang::Decl const& decl) { return decl.getEndLoc(); }

inline ::clang::SourceLocation getBeginLoc(::clang::Expr const& expr) { return expr.getBeginLoc(); }
inline ::clang::SourceLocation getEndLoc(::clang::Expr const& expr) { return expr.getEndLoc(); }

inline ::clang::SourceLocation getBeginLoc(::clang::Stmt const& stmt) { return stmt.getBeginLoc(); }
inline ::clang::SourceLocation getEndLoc(::clang::Stmt const& stmt) { return stmt.getEndLoc(); }
#endif
} // namespace gtclang::clang_compat

#endif // GTCLANG_SUPPORT_CLANGCOMPAT_SOURCELOCATION_H
