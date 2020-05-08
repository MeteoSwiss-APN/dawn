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

SourceLineLoggerProxy::SourceLineLoggerProxy(const SourceLineLoggerProxy& other)
    : logger_(other.logger_), ss_(other.ss_.str()), source_(other.source_), line_(other.line_) {}

SourceLineLoggerProxy::SourceLineLoggerProxy(Logger& logger, const std::string& source, int line)
    : logger_(logger), source_(source), line_(line) {}

SourceLineLoggerProxy::~SourceLineLoggerProxy() { logger_.enqueue(ss_.str(), source_, line_); }

BasicLoggerProxy::BasicLoggerProxy(const BasicLoggerProxy& other)
    : logger_(other.logger_), ss_(other.ss_.str()) {}

BasicLoggerProxy::BasicLoggerProxy(Logger& logger) : logger_(logger) {}

BasicLoggerProxy::~BasicLoggerProxy() { logger_.enqueue(ss_.str()); }

Logger::Logger(Formatter fmt, std::ostream& os) : fmt_(fmt), os_(&os), data_() {}

void Logger::enqueue(const std::string& msg, const std::string& file, int line) {
  data_.push_back(fmt_(msg, file, line));
  *os_ << data_.back();
}

void Logger::enqueue(const std::string& msg) {
  data_.push_back(msg);
  *os_ << data_.back();
}

SourceLineLoggerProxy Logger::operator()(const std::string& source, int line) {
  return SourceLineLoggerProxy(*this, source, line);
}

BasicLoggerProxy Logger::operator()() { return BasicLoggerProxy(*this); }

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

namespace log {
Logger info(makeDefaultFormatter("[INFO]"), std::cout);
Logger warn(makeDefaultFormatter("[WARN]"), std::cerr);
Logger error(makeDefaultFormatter("[ERROR]"), std::cerr);
} // namespace log

} // namespace dawn
