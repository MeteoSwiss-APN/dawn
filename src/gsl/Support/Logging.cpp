//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gsl/Support/Logging.h"
#include "gsl/Support/Assert.h"

namespace gsl {

internal::LoggerProxy::LoggerProxy(LoggingLevel level, std::stringstream& ss, const char* file,
                                   int line)
    : level_(level), ss_(ss), file_(file), line_(line) {}

internal::LoggerProxy::~LoggerProxy() {
  Logger::getSingleton().log(level_, ss_.get().str(), file_, line_);
  ss_.get().str("");
  ss_.get().clear();
}

Logger* Logger::instance_ = nullptr;

Logger::Logger() : logger_(nullptr) {}

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

} // namespace gsl
