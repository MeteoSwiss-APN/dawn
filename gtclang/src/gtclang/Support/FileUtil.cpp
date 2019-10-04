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

#include "gtclang/Support/FileUtil.h"
#include "gtclang/Support/ClangCompat/VirtualFileSystem.h"
#include "clang/Basic/SourceManager.h"

namespace gtclang {

llvm::StringRef getFilename(llvm::StringRef path) {
  return path.substr(path.find_last_of('/') + 1);
}

clang::FileID createInMemoryFile(llvm::StringRef filename, llvm::MemoryBuffer* source,
                                 clang::SourceManager& sources, clang::FileManager& files,
                                 clang_compat::llvm::vfs::InMemoryFileSystem* memFS) {
  memFS->addFileNoOwn(filename, 0, source);
  return sources.createFileID(files.getFile(filename), clang::SourceLocation(),
                              clang::SrcMgr::C_User);
}

} // namespace gtclang
