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

Logger::MessageFormatter makeMessageFormatter(const std::string type) {
  return [type](const std::string& msg, const std::string& file, int line) {
    return "[" + file + ":" + std::to_string(line) + "] " + type + ": " + msg;
  };
}

Logger::DiagnosticFormatter makeDiagnosticFormatter(const std::string type) {
  return [type](const std::string& msg, const std::string& file, int line,
                const std::string& source, SourceLocation loc) {
    std::string diagnostic = "[" + file + ":" + std::to_string(line) + "]" + type + " at " + source;
    std::stringstream ss;
    if(loc.Line >= 0) {
      ss << ":" << loc.Line;
    }
    if(loc.Column >= 0) {
      ss << ":" << loc.Column;
    }
    diagnostic += ss.str() + ": " + msg;
    return diagnostic;
  };
}

MessageProxy::MessageProxy(const MessageProxy& other)
    : logger_(other.logger_), ss_(other.ss_.str()), file_(other.file_), line_(other.line_) {}

MessageProxy::MessageProxy(Logger& logger, const std::string& file, int line)
    : logger_(logger), file_(file), line_(line) {}

MessageProxy::~MessageProxy() { logger_.enqueue(ss_.str(), file_, line_); }

DiagnosticProxy::DiagnosticProxy(const DiagnosticProxy& other)
    : logger_(other.logger_), ss_(other.ss_.str()), file_(other.file_), line_(other.line_),
      source_(other.source_), loc_(other.loc_) {}

DiagnosticProxy::DiagnosticProxy(Logger& logger, const std::string& file, int line,
                                 const std::string& source, SourceLocation loc)
    : logger_(logger), file_(file), line_(line), source_(source), loc_(loc) {}

DiagnosticProxy::~DiagnosticProxy() { logger_.enqueue(ss_.str(), file_, line_, source_, loc_); }

Logger::Logger(MessageFormatter msgFmt, DiagnosticFormatter diagFmt, std::ostream& os, bool show)
    : msgFmt_(msgFmt), diagFmt_(diagFmt), os_(&os), data_(), show_(show) {}

MessageProxy Logger::operator()(const std::string& file, int line) {
  return MessageProxy(*this, file, line);
}

DiagnosticProxy Logger::operator()(const std::string& file, int line, const std::string& source,
                                   SourceLocation loc) {
  return DiagnosticProxy(*this, file, line, source, loc);
}

void Logger::enqueue(const std::string& msg, const std::string& file, int line) {
  data_.push_back(msgFmt_(msg, file, line));
  if(show_)
    *os_ << data_.back();
}

void Logger::enqueue(const std::string& msg, const std::string& file, int line,
                     const std::string& source, SourceLocation loc) {
  data_.push_back(diagFmt_(msg, file, line, source, loc));
  if(show_)
    *os_ << data_.back();
}

std::ostream& Logger::stream() const { return *os_; }
void Logger::stream(std::ostream& os) { os_ = &os; }

Logger::MessageFormatter Logger::messageFormatter() const { return msgFmt_; }
void Logger::messageFormatter(const MessageFormatter& msgFmt) { msgFmt_ = msgFmt; }

Logger::DiagnosticFormatter Logger::diagnosticFormatter() const { return diagFmt_; }
void Logger::diagnosticFormatter(const DiagnosticFormatter& diagFmt) { diagFmt_ = diagFmt; }

void Logger::clear() { data_.clear(); }

void Logger::show() { show_ = true; }
void Logger::hide() { show_ = false; }

// Expose container of messages
Logger::iterator Logger::begin() { return std::begin(data_); }
Logger::iterator Logger::end() { return std::end(data_); }
Logger::const_iterator Logger::begin() const { return std::begin(data_); }
Logger::const_iterator Logger::end() const { return std::end(data_); }
Logger::Container::size_type Logger::size() const { return std::size(data_); }

namespace log {

Logger info(makeMessageFormatter("INFO"), makeDiagnosticFormatter("INFO"), std::cout, false);
Logger warn(makeMessageFormatter("WARNING"), makeDiagnosticFormatter("WARNING"), std::cout, true);
Logger error(makeMessageFormatter("ERROR"), makeDiagnosticFormatter("ERROR"), std::cerr, true);

void setVerbosity(Level level) {
  switch(level) {
  case Level::All:
    info.show();
    warn.show();
    error.show();
    break;
  case Level::Warnings:
    info.hide();
    warn.show();
    error.show();
    break;
  case Level::Errors:
    info.hide();
    warn.hide();
    error.show();
    break;
  case Level::None:
    info.hide();
    warn.hide();
    error.hide();
    break;
  }
}

} // namespace log

} // namespace dawn
