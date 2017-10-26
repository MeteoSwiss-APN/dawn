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

  /// @brief Get full path of `filename` or abort if file was not found in `{unittest directory}/relativePath`
  std::string getUnittestFile(llvm::StringRef relativePath, llvm::StringRef filename) const;

  /// @brief Get full path of `filename` or abort if file was not found in `{integrationtest directory}/relativePath`
  std::string getIntegrationtestFile(llvm::StringRef relativePath, llvm::StringRef filename) const;

  /// @brief Get full path of the unittest source directory
  std::string getIntegrationtestPath() const;

  /// @brief Get full path of the integrationtest source directory
  std::string getUnittestPath() const;

  /// @brief Creates all the required direcotries such that 'fullpath' is a valid location
  void createRequiredFolders(llvm::StringRef fullpath) const;

private:
  /// @brief Get full path of `filename` or abort if file was not found in `filePath`
  std::string getFile(llvm::StringRef filePath, llvm::StringRef filename) const;
};

} // namespace gtclang

#endif
