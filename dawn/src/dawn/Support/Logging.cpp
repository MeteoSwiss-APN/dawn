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
#include "dawn/Support/FileSystem.h"
#include "dawn/Support/Format.h"
#include <chrono>
#include <iostream>

namespace dawn {

Logger::Formatter makeDefaultFormatter(const std::string prefix) {
  return [prefix](const std::string& msg, const std::string& file, int line) {
    return prefix + " " + "[" + file + ":" + std::to_string(line) + "] " + msg;
  };
}

LoggerProxy::LoggerProxy(const LoggerProxy& other)
    : logger_(other.logger_), ss_(other.ss_.str()), source_(other.source_), line_(other.line_) {}

LoggerProxy::LoggerProxy(Logger& logger, const std::string& source, int line)
    : logger_(logger), source_(source), line_(line) {}

LoggerProxy::~LoggerProxy() { logger_.enqueue(ss_.str(), source_, line_); }

Logger::Logger(Formatter fmt, std::ostream& os) : fmt_(fmt), os_(&os), data_() {}

void Logger::enqueue(const std::string& msg, const std::string& file, int line) {
  data_.push_back(fmt_(msg, file, line));
  *os_ << data_.back();
}

LoggerProxy Logger::operator()(const std::string& source, int line) {
  return LoggerProxy(*this, source, line);
}

// Get and set ostream
std::ostream& Logger::stream() const { return *os_; }
void Logger::stream(std::ostream& os) { os_ = &os; }

// Get and set Formatter
Logger::Formatter Logger::formatter() const { return fmt_; }
void Logger::formatter(const Formatter& fmt) { fmt_ = fmt; }

// Reset storage
void Logger::clear() { data_.clear(); }

// Expose container of messages
Logger::iterator Logger::begin() { return std::begin(data_); }
Logger::iterator Logger::end() { return std::end(data_); }
Logger::const_iterator Logger::begin() const { return std::begin(data_); }
Logger::const_iterator Logger::end() const { return std::end(data_); }
Logger::MessageContainer::size_type Logger::size() const { return std::size(data_); }

Logger info(makeDefaultFormatter("[INFO]"), std::cout);
Logger warn(makeDefaultFormatter("[WARN]"), std::cout);
Logger error(makeDefaultFormatter("[ERROR"), std::cerr);

} // namespace dawn
