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

namespace gtclang {

dawn::Logger::MessageFormatter makeGTClangMessageFormatter(const std::string& prefix) {
  return [prefix](const std::string& message, const std::string& file, int line) -> std::string {
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

    ss << prefix;
    ss << prefix << " [" << file << ":" << line << "] " << message << "\n";

    return ss.str();
  };
}

dawn::Logger::DiagnosticFormatter makeGTClangDiagnosticFormatter(const std::string& prefix) {
  return [prefix](const std::string& message, const std::string& file, int line,
                  const std::string& source, dawn::SourceLocation loc) -> std::string {
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

    ss << prefix;
    ss << prefix << " [" << file << ":" << line << "] " << source;

    if(loc.Line) {
      ss << ":" << loc.Line;
    }
    if(loc.Column) {
      ss << ":" << loc.Column;
    }
    ss << " : " << message << "\n";

    return ss.str();
  };
}

} // namespace gtclang
