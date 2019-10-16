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

#ifndef GTCLANG_SUPPORT_CLANGCOMPAT_FILESYSTEM_H
#define GTCLANG_SUPPORT_CLANGCOMPAT_FILESYSTEM_H

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"

namespace gtclang::clang_compat {
#if LLVM_VERSION_MAJOR < 7
namespace llvm::sys::fs::OpenFlags {
static constexpr ::llvm::sys::fs::OpenFlags OF_Text = ::llvm::sys::fs::OpenFlags::F_Text;
}
#else
namespace llvm::sys::fs::OpenFlags {
static constexpr ::llvm::sys::fs::OpenFlags OF_Text = ::llvm::sys::fs::OpenFlags::OF_Text;
}
#endif
} // namespace gtclang::clang_compat

#endif // GTCLANG_SUPPORT_CLANGCOMPAT_FILESYSTEM_H
