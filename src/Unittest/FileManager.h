//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _     _ _              _            _
//                        (_)   | | |            | |          | |
//               __ _ _ __ _  __| | |_ ___   ___ | |___    ___| | __ _ _ __   __ _
//              / _` | '__| |/ _` | __/ _ \ / _ \| / __|  / __| |/ _` | '_ \ / _` |
//             | (_| | |  | | (_| | || (_) | (_) | \__ \ | (__| | (_| | | | | (_| |
//              \__, |_|  |_|\__,_|\__\___/ \___/|_|___/  \___|_|\__,_|_| |_|\__, |
//               __/ |                                                        __/ |
//              |___/                                                        |___/
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
  std::string dataPath_;

public:
  FileManager();

  /// @brief Get full path of `filename` or abort if file was not found in `dataPath()` directory
  std::string getFile(llvm::StringRef filename) const;

  /// @brief Path of the unittest data files
  const std::string& dataPath() const { return dataPath_; }
};

} // namespace gtclang

#endif
