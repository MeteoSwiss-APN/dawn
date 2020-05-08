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

#include "dawn/Support/Logger.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <sstream>

using namespace dawn;

namespace {

TEST(Logger, construction) {
  std::ostringstream buffer;
  Logger log(makeMessageFormatter(), makeDiagnosticFormatter(), buffer);
  EXPECT_EQ(log.size(), 0);
}

TEST(Logger, message_formatting) {
  std::ostringstream buffer;
  Logger log(makeMessageFormatter("LOG"), makeDiagnosticFormatter(), buffer);
  log("TestLogger.cpp", 42) << "Message";
  EXPECT_EQ(buffer.str(), "[TestLogger.cpp:42] LOG: Message\n");
}

TEST(Logger, diagnostic_formatting) {
  std::ostringstream buffer;
  Logger log(makeMessageFormatter(), makeDiagnosticFormatter("LOG"), buffer);
  log("TestLogger.cpp", 42, "test.input", SourceLocation(42, 4)) << "Message";
  EXPECT_EQ(buffer.str(), "[TestLogger.cpp:42] test.input:42:4: LOG: Message\n");
}

TEST(Logger, stream) {
  std::ostringstream buffer;
  Logger log(makeMessageFormatter(), makeDiagnosticFormatter(), buffer);
  std::ostringstream anotherBuffer;
  log.stream(anotherBuffer);
  log("TestLogger.cpp", 42) << "Message";
  EXPECT_EQ(buffer.str(), "");
  EXPECT_NE(anotherBuffer.str(), "");
}

TEST(Logger, custom_MessageFormatter) {
  {
    std::ostringstream buffer;
    // Initialize with standard formatter
    Logger log(makeMessageFormatter(), makeDiagnosticFormatter(), buffer);
    // Replace formatter
    log.messageFormatter([](const std::string& msg, const std::string& file, int line) {
      return std::string(std::rbegin(msg), std::rend(msg));
    });
    log("TestLogger.cpp", 42) << "message";
    const std::string firstMessage = *log.begin();
    EXPECT_EQ(firstMessage[0], 'e');
    EXPECT_EQ(firstMessage[6], 'm');
  }
  {
    std::ostringstream buffer;
    Logger log([](const std::string& msg, const std::string& file,
                  int line) { return std::string(std::rbegin(msg), std::rend(msg)); },
               makeDiagnosticFormatter(), buffer);
    log("TestLogger.cpp", 42) << "message";
    const std::string firstMessage = *log.begin();
    EXPECT_EQ(firstMessage[0], 'e');
    EXPECT_EQ(firstMessage[6], 'm');
  }
}

TEST(Logger, custom_DiagnosticFormatter) {
  {
    std::ostringstream buffer;
    // Initialize with standard formatter
    Logger log(makeMessageFormatter(), makeDiagnosticFormatter(), buffer);
    // Replace formatter
    log.diagnosticFormatter(
        [](const std::string& msg, const std::string& file, int line, const std::string& source,
           SourceLocation loc) { return std::string(std::rbegin(msg), std::rend(msg)); });
    log("TestLogger.cpp", 42, "", SourceLocation()) << "message";
    const std::string firstMessage = *log.begin();
    EXPECT_EQ(firstMessage[0], 'e');
    EXPECT_EQ(firstMessage[6], 'm');
  }
  {
    std::ostringstream buffer;
    Logger log(makeMessageFormatter(),
               [](const std::string& msg, const std::string& file, int line,
                  const std::string& source,
                  SourceLocation loc) { return std::string(std::rbegin(msg), std::rend(msg)); },
               buffer);
    log("TestLogger.cpp", 42, "", SourceLocation()) << "message";
    const std::string firstMessage = *log.begin();
    EXPECT_EQ(firstMessage[0], 'e');
    EXPECT_EQ(firstMessage[6], 'm');
  }
}

TEST(Logger, size_and_clear) {
  std::ostringstream buffer;
  Logger log(makeMessageFormatter(), makeDiagnosticFormatter(), buffer);
  log("TestLogger.cpp", 42) << "A message\n";
  log("TestLogger.cpp", 42) << "Another message\n";
  EXPECT_EQ(log.size(), 2);
  log.clear();
  EXPECT_EQ(log.size(), 0);
}

TEST(Logger, show_and_hide) {
  std::ostringstream buffer;
  Logger log(makeMessageFormatter(), makeDiagnosticFormatter(), buffer, false);
  log("TestLogger.cpp", 42) << "A message\n";
  EXPECT_EQ(buffer.str(), "");
  EXPECT_EQ(log.size(), 1);
  log.show();
  log("TestLogger.cpp", 42) << "A message\n";
  EXPECT_NE(buffer.str(), "");
  EXPECT_EQ(log.size(), 2);
}

TEST(Logger, iterate) {
  std::ostringstream buffer;
  Logger log([](const std::string& msg, const std::string& file, int line) { return msg; },
             makeDiagnosticFormatter(), buffer);
  log("TestLogger.cpp", 42) << "A message";
  log("TestLogger.cpp", 42) << "Another message";
  auto iter = log.begin();
  EXPECT_EQ(iter->size(), 9);
  ++iter;
  EXPECT_EQ(iter->size(), 15);
}

} // namespace
