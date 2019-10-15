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

namespace gtclang {
namespace clang_compat {
namespace CompilerInvocation {
#if CLANG_VERSION_MAJOR < 9
inline bool CreateFromArgs(clang::CompilerInvocation& Res, llvm::opt::ArgStringList& ccArgs,
                           clang::DiagnosticsEngine& Diags) {
  return clang::CompilerInvocation::CreateFromArgs(
      Res, const_cast<const char**>(ccArgs.data()),
      const_cast<const char**>(ccArgs.data()) + ccArgs.size(), Diags);
}

#else
inline bool CreateFromArgs(clang::CompilerInvocation& Res, llvm::opt::ArgStringList& ccArgs,
                           clang::DiagnosticsEngine& Diags) {
  const char* args_tmp[ccArgs.size()];
  std::size_t i = 0;
  for(auto a : ccArgs) {
    args_tmp[i] = a;
    ++i;
  }
  llvm::ArrayRef<const char*> argsArrayRef(args_tmp, ccArgs.size());
  return clang::CompilerInvocation::CreateFromArgs(Res, argsArrayRef, Diags);
}

#endif
} // namespace CompilerInvocation
} // namespace clang_compat
} // namespace gtclang

#endif // GTCLANG_SUPPORT_CLANGCOMPAT_COMPILER_INVOCATION_H
