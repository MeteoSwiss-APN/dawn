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

std::string FileManager::getUnittestPath() const { return GTCLANG_UNITTEST_DATAPATH; }

std::string FileManager::getIntegrationtestPath() const { return GTCLANG_INTEGRATIONTEST_DATAPATH; }

std::string FileManager::getUnittestFile(llvm::StringRef relativePath,
                                         llvm::StringRef filename) const {
  return getFile(getUnittestPath() + "/" + std::string(relativePath), filename);
}

std::string FileManager::getIntegrationtestFile(llvm::StringRef relativePath,
                                                llvm::StringRef filename) const {
  return getFile(getIntegrationtestPath() + "/" + std::string(relativePath), filename);
}

std::string FileManager::getFile(llvm::StringRef filePath, llvm::StringRef filename) const {

  using namespace llvm;
  std::string path = std::string(filePath) + "/" + std::string(filename);

  if(!sys::fs::exists(path)) {
    errs().changeColor(llvm::raw_ostream::RED, true) << "FATAL ERROR ";
    errs().resetColor() << ": file '" << path << "' not found!\n";
    std::abort();
  }

  return path;
}

void FileManager::createRequiredFolders(llvm::StringRef fullpath) const {
  llvm::sys::fs::create_directories(fullpath);
}

} // namespace gtclang
