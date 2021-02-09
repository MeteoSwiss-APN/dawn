#pragma once

#include "clang/Basic/FileManager.h"
#include "clang/Basic/Version.h"

namespace dawn::clang_compat::FileUtil {
#if CLANG_VERSION_MAJOR < 10
inline const clang::FileEntry* getFile(clang::FileManager& files, ::llvm::StringRef filename) {
  return files.getFile(filename);
}
#else
inline const clang::FileEntry* getFile(clang::FileManager& files, ::llvm::StringRef filename) {
  return files.getFile(filename).get(); // maybe check for error
}
#endif
} // namespace dawn::clang_compat::FileUtil
