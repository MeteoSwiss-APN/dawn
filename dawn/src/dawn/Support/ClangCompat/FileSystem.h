#pragma once

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"

namespace dawn::clang_compat {
#if LLVM_VERSION_MAJOR < 7
namespace llvm::sys::fs::OpenFlags {
static constexpr ::llvm::sys::fs::OpenFlags OF_Text = ::llvm::sys::fs::OpenFlags::F_Text;
}
#else
namespace llvm::sys::fs::OpenFlags {
static constexpr ::llvm::sys::fs::OpenFlags OF_Text = ::llvm::sys::fs::OpenFlags::OF_Text;
}
#endif
} // namespace dawn::clang_compat
