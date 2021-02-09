#pragma once

#include "clang/AST/Expr.h"
#include "clang/Basic/Version.h"

namespace dawn::clang_compat::Expr {
#if CLANG_VERSION_MAJOR < 8
using EvalResultInt = ::llvm::APSInt;
inline int64_t getInt(EvalResultInt const& res) { return res.getExtValue(); }

#else
using EvalResultInt = ::clang::Expr::EvalResult;
inline int64_t getInt(EvalResultInt const& res) { return res.Val.getInt().getExtValue(); }

#endif
} // namespace dawn::clang_compat::Expr
