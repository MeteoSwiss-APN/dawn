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

#include "gtclang/Support/Logger.h"
#include "dawn/Support/Format.h"
#include <chrono>
#include <sstream>

namespace {
enum Code {
  FG_RED = 31,
  FG_GREEN = 32,
  FG_BLUE = 34,
  FG_DEFAULT = 39,

  BG_RED = 41,
  BG_GREEN = 42,
  BG_BLUE = 44,
  BG_DEFAULT = 49
};

class Change {
  Code code;

public:
  Change(Code _code) : code(_code) {}

  friend std::ostream& operator<<(std::ostream& os, const Change& chng) {
    return os << "\033[" << chng.code << "m";
  }
};

Change red(FG_RED);
Change green(FG_GREEN);
Change blue(FG_BLUE);
Change reset(FG_DEFAULT);
} // namespace

namespace gtclang {

dawn::Logger::Formatter makeGTClangFormatter(dawn::LoggingLevel level) {
  return [level](const std::string& message, const std::string& file, int line) -> std::string {
    // Get current date-time (up to ms accuracy)
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    auto now_ms = now.time_since_epoch();
    auto now_sec = std::chrono::duration_cast<std::chrono::seconds>(now_ms);
    auto tm_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now_ms - now_sec);

    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    struct tm* localTime = std::localtime(&currentTime);

    auto timeStr = dawn::format("%02i:%02i:%02i.%03i", localTime->tm_hour, localTime->tm_min,
                                localTime->tm_sec, tm_ms.count());

    std::stringstream ss;
    ss << "[" << timeStr << "] ";

    switch(level) {
    case dawn::LoggingLevel::Info:
      ss << "[INFO]";
      break;
    case dawn::LoggingLevel::Warning:
      ss << red << "[WARN]" << reset;
      break;
    case dawn::LoggingLevel::Error:
      ss << red << "[ERROR]" << reset;
      break;
    case dawn::LoggingLevel::Fatal:
      ss << red << "[FATAL]" << reset;
      break;
    }

    ss << " [" << file << ":" << line << "] " << message << "\n";

    return ss.str();
  };
}

} // namespace gtclang
