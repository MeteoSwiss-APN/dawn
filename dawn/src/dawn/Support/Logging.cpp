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

#include "dawn/Support/Logging.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Filesystem.h"
#include "dawn/Support/Format.h"
#include <chrono>
#include <iostream>

namespace dawn {

internal::LoggerProxy::LoggerProxy(LoggingLevel level, std::stringstream& ss, const char* file,
                                   int line)
    : level_(level), ss_(ss), file_(file), line_(line) {}

internal::LoggerProxy::~LoggerProxy() {
  Logger::getSingleton().log(level_, ss_.get().str(), file_, line_);
  ss_.get().str("");
  ss_.get().clear();
}

Logger* Logger::instance_ = nullptr;

Logger::Logger() { registerLogger(new DefaultLogger); }
Logger::~Logger() { delete logger_; }

void Logger::registerLogger(LoggerInterface* logger) { logger_ = logger; }

LoggerInterface* Logger::getLogger() { return logger_; }

internal::LoggerProxy Logger::logInfo(const char* file, int line) {
  return internal::LoggerProxy(LoggingLevel::Info, ss_, file, line);
}

internal::LoggerProxy Logger::logWarning(const char* file, int line) {
  return internal::LoggerProxy(LoggingLevel::Warning, ss_, file, line);
}

internal::LoggerProxy Logger::logError(const char* file, int line) {
  return internal::LoggerProxy(LoggingLevel::Error, ss_, file, line);
}

internal::LoggerProxy Logger::logFatal(const char* file, int line) {
  return internal::LoggerProxy(LoggingLevel::Fatal, ss_, file, line);
}

void Logger::log(LoggingLevel level, const std::string& message, const char* file, int line) {
  if(logger_ != nullptr) {
    logger_->log(level, message, file, line);
  }
}

Logger& Logger::getSingleton() {
  if(instance_ == nullptr) {
    instance_ = new Logger;
  }
  return *instance_;
}

void DefaultLogger::log(LoggingLevel level, const std::string& message, const char* file,
                        int line) {
  fs::path filePath(file);
  const std::string fileStr = filePath.stem();

  std::stringstream ss;

  // Get current date-time (up to ms accuracy)
  std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
  auto now_ms = now.time_since_epoch();
  auto now_sec = std::chrono::duration_cast<std::chrono::seconds>(now_ms);
  auto tm_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now_ms - now_sec);

  std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
  struct tm* localTime = std::localtime(&currentTime);

  auto timeStr = dawn::format("%02i:%02i:%02i.%03i", localTime->tm_hour, localTime->tm_min,
                              localTime->tm_sec, tm_ms.count());

  ss << "[" << timeStr << "] [" << fileStr << ":" << line << "] [";

  switch(level) {
  case LoggingLevel::Info:
    ss << "INFO";
    break;
  case LoggingLevel::Warning:
    ss << "WARN";
    break;
  case LoggingLevel::Error:
    ss << "ERROR";
    break;
  case LoggingLevel::Fatal:
    ss << "FATAL";
    break;
  }

  ss << "] " << message << "\n";

  switch(level) {
  case LoggingLevel::Info:
    std::cout << ss.str();
    break;
  case LoggingLevel::Warning:
    std::cout << ss.str();
    break;
  case LoggingLevel::Error:
    std::cerr << ss.str();
    break;
  case LoggingLevel::Fatal:
    std::cerr << ss.str();
    break;
  }
}

} // namespace dawn
