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

#ifndef GTCLANG_SUPPORT_FILEUTIL_H
#define GTCLANG_SUPPORT_FILEUTIL_H

#include "gtclang/Support/ClangCompat/VirtualFileSystem.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringRef.h"

namespace gtclang {

/// @brief Extract the filename from `path`
///
/// This will only work on UNIX like platforms.
///
/// @ingroup support
extern llvm::StringRef getFilename(llvm::StringRef path);

/// @brief Create a new "in-memory" file
/// @ingroup support
extern clang::FileID createInMemoryFile(llvm::StringRef filename, llvm::MemoryBuffer* source,
                                        clang::SourceManager& sources, clang::FileManager& files,
                                        clang_compat::llvm::vfs::InMemoryFileSystem* memFS);

} // namespace gtclang

#endif
