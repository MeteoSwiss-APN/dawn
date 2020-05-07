//===--------------------------------------------------------------------------------*- C++ -*-===//
//                          _
//                         | |
//                       __| | __ ___      ___ ___
//                      / _` |/ _` \ \ /\ / / '_  |
//                     | (_| | (_| |\ V  V /| | | |
//                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

// TODO Name this "Logger"
#include "dawn/Support/Logging.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <sstream>

using namespace dawn;

namespace {

TEST(Logger, Construction) {
  std::ostringstream buffer;
  Logger log(makeDefaultFormatter("[LOG]"), buffer);
  EXPECT_EQ(log.size(), 0);
}

TEST(Logger, size) {
  std::ostringstream buffer;
  Logger log(makeDefaultFormatter("[LOG]"), buffer);
  log(__FILE__, __LINE__) << "A message\n";
  log(__FILE__, __LINE__) << "Another message\n";
  EXPECT_EQ(log.size(), 2);
}

TEST(Logger, CustomFormatter) {
  {
    std::ostringstream buffer;
    // Initialize with standard formatter
    Logger log(makeDefaultFormatter(""), buffer);
    // Replace formatter
    log.formatter([](const std::string& msg, const std::string& file, int line) {
      return std::string(std::rbegin(msg), std::rend(msg));
    });
    log(__FILE__, __LINE__) << "message";
    const std::string firstMessage = *log.begin();
    EXPECT_EQ(firstMessage[0], 'e');
    EXPECT_EQ(firstMessage[6], 'm');
  }
  {
    std::ostringstream buffer;
    Logger log([](const std::string& msg, const std::string& file,
                  int line) { return std::string(std::rbegin(msg), std::rend(msg)); },
               buffer);
    log(__FILE__, __LINE__) << "message";
    const std::string firstMessage = *log.begin();
    EXPECT_EQ(firstMessage[0], 'e');
    EXPECT_EQ(firstMessage[6], 'm');
  }
}

TEST(Logger, iterate) {
  std::ostringstream buffer;
  Logger log(makeDefaultFormatter("[LOG]"), buffer);
  log(__FILE__, __LINE__) << "A message";
  log(__FILE__, __LINE__) << "Another message";
  auto iter = log.begin();
  EXPECT_EQ(iter->size(), 9);
  ++iter;
  EXPECT_EQ(iter->size(), 15);
}

} // namespace
