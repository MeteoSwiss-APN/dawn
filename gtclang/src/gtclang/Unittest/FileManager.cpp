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

#include "gtclang/Unittest/FileManager.h"
#include "gtclang/Support/Config.h"
#include "gtclang/Unittest/Config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace gtclang {

std::string FileManager::getUnittestFile(llvm::StringRef relativePath,
                                         llvm::StringRef filename) const {
  std::string fullpath = std::string(GTCLANG_UNITTEST_DATAPATH) + "/" + std::string(relativePath);
  createRequiredFolders(fullpath);

  return fullpath + "/" + std::string(filename);
}

std::string FileManager::getIntegrationtestFile(llvm::StringRef relativePath,
                                                llvm::StringRef filename) const {
  std::string fullpath =
      std::string(GTCLANG_INTEGRATIONTEST_DATAPATH) + "/" + std::string(relativePath);
  createRequiredFolders(fullpath);
  return fullpath + "/" + std::string(filename);
}

void FileManager::createRequiredFolders(llvm::StringRef fullpath) const {
  llvm::sys::fs::create_directories(fullpath);

  if(!llvm::sys::fs::exists(fullpath)) {
    llvm::errs().changeColor(llvm::raw_ostream::RED, true) << "FATAL ERROR ";
    llvm::errs().resetColor() << ": could not generate " << fullpath << "' !\n";
    std::abort();
  }
}

} // namespace gtclang
