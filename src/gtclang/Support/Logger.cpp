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
#include "llvm/Support/raw_ostream.h"
#include <chrono>

namespace gtclang {

void Logger::log(dawn::LoggingLevel level, const std::string& message, const char* file, int line) {
  using namespace llvm;

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

  outs() << "[" << timeStr << "] ";

  switch(level) {
  case dawn::LoggingLevel::Info:
    outs() << "[INFO]";
    break;
  case dawn::LoggingLevel::Warning:
    outs().changeColor(raw_ostream::MAGENTA, true) << "[WARN]";
    outs().resetColor();
    break;
  case dawn::LoggingLevel::Error:
    outs().changeColor(raw_ostream::RED, true) << "[ERROR]";
    outs().resetColor();
    break;
  case dawn::LoggingLevel::Fatal:
    outs().changeColor(raw_ostream::RED, true) << "[FATAL]";
    outs().resetColor();
    break;
  }

  outs() << " [" << fileStr << ":" << line << "] " << message << "\n";
}

} // namespace gtclang
