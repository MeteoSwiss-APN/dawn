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

#ifndef GTCLANG_UNITTEST_FLAGMANAGER_H
#define GTCLANG_UNITTEST_FLAGMANAGER_H

#include "gtclang/Unittest/Config.h"
#include <string>
#include <vector>

namespace gtclang {

/// @brief Handle flags for the gtclang tool
/// @ingroup unittest
class FlagManager {
  std::string includes_;

public:
  FlagManager();

  /// @brief Get include directories
  std::string getInclude() const;

  /// @brief Get default flags
  std::vector<std::string> getDefaultFlags() const;
};

} // namespace gtclang

#endif
