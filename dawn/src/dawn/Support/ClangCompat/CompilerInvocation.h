#pragma once

#include "clang/Basic/Version.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "llvm/Option/Option.h"

namespace dawn::clang_compat::CompilerInvocation {
#if CLANG_VERSION_MAJOR < 10
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
} // namespace dawn::clang_compat::CompilerInvocation
