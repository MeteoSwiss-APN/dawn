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

#ifndef GTCLANG_UNITTEST_FILEMANAGER_H
#define GTCLANG_UNITTEST_FILEMANAGER_H

#include "llvm/ADT/StringRef.h"
#include <string>

namespace gtclang {

/// @brief Handle unittest data files
/// @ingroup unittest
class FileManager {

public:
  FileManager() = default;

  /// @brief Get full path of `filename` or abort if required directories could not be created
  /// `{unittest directory}/relativePath`
  std::string getUnittestFile(llvm::StringRef relativePath, llvm::StringRef filename) const;

  /// @brief Get full path of `filename` or abort if required directories could not be created
  /// `{integrationtest directory}/relativePath`
  std::string getIntegrationtestFile(llvm::StringRef relativePath, llvm::StringRef filename) const;

private:
  /// @brief Creates all the required direcotries such that 'fullpath' is a valid location
  void createRequiredFolders(llvm::StringRef fullpath) const;
};

} // namespace gtclang

#endif
