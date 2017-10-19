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

#ifndef GTCLANG_UNITTEST_UNITTESTENVIRONMENT_H
#define GTCLANG_UNITTEST_UNITTESTENVIRONMENT_H

#include "gtclang/Unittest/FileManager.h"
#include "gtclang/Unittest/FlagManager.h"
#include <gtest/gtest.h>
#include <string>

namespace gtclang {

/// @brief Global unittest environment (Singleton)
/// @ingroup unittest
class UnittestEnvironment : public testing::Environment {
  static UnittestEnvironment* instance_;
  FileManager fileManager_;
  FlagManager flagManager_;

public:
  virtual ~UnittestEnvironment();

  /// @brief Set up the environment.
  virtual void SetUp();

  /// @brief Tear down the environment.
  virtual void TearDown();

  /// @brief Name of the current test-case
  /// @return Name of the current test-case or an empty string if called outside a test
  std::string testCaseName() const;

  /// @brief Name of the current test
  /// @return Name of the current test or an empty string if called outside a test
  std::string testName() const;

  /// @brief Get singleton instance
  static UnittestEnvironment& getSingleton();

  /// @brief Get FileManager
  const FileManager& getFileManager() { return fileManager_; }

  /// @brief Get FlagManager
  const FlagManager& getFlagManager() { return flagManager_; }
};

} // namespace gtclang

#endif
