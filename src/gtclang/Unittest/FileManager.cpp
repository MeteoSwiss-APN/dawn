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

FileManager::FileManager() : dataPath_(GTCLANG_UNITTEST_DATAPATH) {}

void FileManager::setKind(testKind kind) {
  if(kind == testKind::unittest) {
    dataPath_ = GTCLANG_UNITTEST_DATAPATH;
  }
  if(kind == testKind::integrationtet) {
    dataPath_ = GTCLANG_INTEGRATIONTEST_DATAPATH;
  }
}

std::string FileManager::getFile(llvm::StringRef filename) const {
  using namespace llvm;
  Twine path = llvm::StringRef(dataPath_) + "/" + filename;

  if(!sys::fs::exists(path)) {
    errs().changeColor(llvm::raw_ostream::RED, true) << "FATAL ERROR ";
    errs().resetColor() << ": file '" << path.str() << "' not found!\n";
    std::abort();
  }

  return path.str();
}

} // namespace gtclang
