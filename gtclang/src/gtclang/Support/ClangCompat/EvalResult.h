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

#ifndef GTCLANG_SUPPORT_CLANGCOMPAT_EVALRESULT_H
#define GTCLANG_SUPPORT_CLANGCOMPAT_EVALRESULT_H

#include "clang/AST/Expr.h"
#include "clang/Basic/Version.h"

namespace gtclang::clang_compat::Expr {
#if CLANG_VERSION_MAJOR < 8
using EvalResultInt = ::llvm::APSInt;
inline int64_t getInt(EvalResultInt const& res) { return res.getExtValue(); }

#else
using EvalResultInt = ::clang::Expr::EvalResult;
inline int64_t getInt(EvalResultInt const& res) { return res.Val.getInt().getExtValue(); }

#endif
} // namespace gtclang::clang_compat::Expr

#endif // GTCLANG_SUPPORT_CLANGCOMPAT_EVALRESULT_H
