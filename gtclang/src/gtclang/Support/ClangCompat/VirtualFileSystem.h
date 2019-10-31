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

#ifndef GTCLANG_SUPPORT_CLANGCOMPAT_VIRTUALFILESYSTEM_H
#define GTCLANG_SUPPORT_CLANGCOMPAT_VIRTUALFILESYSTEM_H

#include "clang/Basic/Version.h"

#if CLANG_VERSION_MAJOR < 8
#include "clang/Basic/VirtualFileSystem.h"
#else
#include "llvm/Support/VirtualFileSystem.h"
#endif

namespace gtclang::clang_compat {
#if CLANG_VERSION_MAJOR < 8

namespace llvm::vfs {
using InMemoryFileSystem = ::clang::vfs::InMemoryFileSystem;
}
#else
namespace llvm::vfs {
using InMemoryFileSystem = ::llvm::vfs::InMemoryFileSystem;
} // namespace llvm::vfs
#endif

} // namespace gtclang::clang_compat

#endif // GTCLANG_SUPPORT_CLANGCOMPAT_VIRTUALFILESYSTEM_H
