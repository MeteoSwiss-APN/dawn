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

#ifndef GTCLANG_SUPPORT_CLANGCOMPAT_FILE_UTIL_H
#define GTCLANG_SUPPORT_CLANGCOMPAT_FILE_UTIL_H

#include "clang/Basic/FileManager.h"
#include "clang/Basic/Version.h"

namespace gtclang::clang_compat::FileUtil {
#if CLANG_VERSION_MAJOR < 9
inline const clang::FileEntry* getFile(clang::FileManager& files, ::llvm::StringRef filename) {
  return files.getFile(filename);
}
#else
inline const clang::FileEntry* getFile(clang::FileManager& files, ::llvm::StringRef filename) {
  return files.getFile(filename).get(); // maybe check for error
}
#endif
} // namespace gtclang::clang_compat::FileUtil

#endif // GTCLANG_SUPPORT_CLANGCOMPAT_FILE_UTIL_H
