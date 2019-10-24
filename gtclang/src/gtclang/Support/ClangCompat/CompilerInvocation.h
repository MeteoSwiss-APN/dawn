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

#ifndef GTCLANG_SUPPORT_CLANGCOMPAT_COMPILER_INVOCATION_H
#define GTCLANG_SUPPORT_CLANGCOMPAT_COMPILER_INVOCATION_H

#include "clang/Basic/Version.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "llvm/Option/Option.h"

namespace gtclang::clang_compat::CompilerInvocation {
#if CLANG_VERSION_MAJOR <= 9
inline bool CreateFromArgs(clang::CompilerInvocation& Res, llvm::opt::ArgStringList& ccArgs,
                           clang::DiagnosticsEngine& Diags) {
  return clang::CompilerInvocation::CreateFromArgs(
      Res, const_cast<const char**>(ccArgs.data()),
      const_cast<const char**>(ccArgs.data()) + ccArgs.size(), Diags);
}
#else
inline bool CreateFromArgs(clang::CompilerInvocation& Res, llvm::opt::ArgStringList& ccArgs,
                           clang::DiagnosticsEngine& Diags) {
  return clang::CompilerInvocation::CreateFromArgs(Res, ccArgs, Diags);
}
#endif
} // namespace gtclang::clang_compat::CompilerInvocation

#endif // GTCLANG_SUPPORT_CLANGCOMPAT_COMPILER_INVOCATION_H
