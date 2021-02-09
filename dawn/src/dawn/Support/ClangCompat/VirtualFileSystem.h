#pragma once

#include "clang/Basic/Version.h"

#if CLANG_VERSION_MAJOR < 8
#include "clang/Basic/VirtualFileSystem.h"
#else
#include "llvm/Support/VirtualFileSystem.h"
#endif

namespace dawn::clang_compat {
#if CLANG_VERSION_MAJOR < 8

namespace llvm::vfs {
using InMemoryFileSystem = ::clang::vfs::InMemoryFileSystem;
}
#else
namespace llvm::vfs {
using InMemoryFileSystem = ::llvm::vfs::InMemoryFileSystem;
} // namespace llvm::vfs
#endif

} // namespace dawn::clang_compat
