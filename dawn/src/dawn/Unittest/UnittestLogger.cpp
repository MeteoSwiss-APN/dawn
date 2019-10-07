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

#include "dawn/Unittest/UnittestLogger.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/StringRef.h"
#include <array>
#include <chrono>
#include <iostream>

namespace dawn {

void UnittestLogger::log(LoggingLevel level, const std::string& message, const char* file,
                         int line) {
  StringRef fileStr(file);
  fileStr = fileStr.substr(fileStr.find_last_of('/') + 1);

  // Get current date-time (up to ms accuracy)
  std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
  auto now_ms = now.time_since_epoch();
  auto now_sec = std::chrono::duration_cast<std::chrono::seconds>(now_ms);
  auto tm_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now_ms - now_sec);

  std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
  struct tm* localTime = std::localtime(&currentTime);

  auto timeStr = dawn::format("%02i:%02i:%02i.%03i", localTime->tm_hour, localTime->tm_min,
                              localTime->tm_sec, tm_ms.count());

  std::cout << "[" << timeStr << "] [" << fileStr << ":" << line << "] [";

  switch(level) {
  case LoggingLevel::Info:
    std::cout << "INFO";
    break;
  case LoggingLevel::Warning:
    std::cout << "WARN";
    break;
  case LoggingLevel::Error:
    std::cout << "ERROR";
    break;
  case LoggingLevel::Fatal:
    std::cout << "FATAL";
    break;
  }

  std::cout << "] " << message << "\n";
}

} // namespace dawn
