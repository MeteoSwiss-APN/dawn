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

#pragma once

#include "dawn/Support/SourceLocation.h"
#include <functional>
#include <iostream>
#include <list>
#include <sstream>
#include <stack>
#include <string>
#include <tuple>

namespace dawn {

class Logger;

/// @brief Proxy for logging messages.
class MessageProxy {
public:
  MessageProxy(Logger& logger, const std::string& file, int line);
  MessageProxy(const MessageProxy& other);
  ~MessageProxy();

  template <typename Streamable>
  MessageProxy& operator<<(Streamable&& obj) {
    ss_ << obj;
    return *this;
  }

private:
  Logger& logger_;
  std::stringstream ss_;
  const std::string file_;
  const int line_;
};

/// @brief Proxy for logging diagnostics.
class DiagnosticProxy {
public:
  DiagnosticProxy(Logger& logger, const std::string& file, int line, const std::string& source,
                  SourceLocation loc);
  DiagnosticProxy(const DiagnosticProxy& other);
  ~DiagnosticProxy();

  template <typename Streamable>
  DiagnosticProxy& operator<<(Streamable&& obj) {
    ss_ << obj;
    return *this;
  }

private:
  Logger& logger_;
  std::stringstream ss_;
  const std::string file_;
  const int line_;
  const std::string source_;
  const SourceLocation loc_;
};

/// @brief Logging interface
/// @ingroup support
class Logger {
public:
  using Container = std::list<std::string>;
  using MessageFormatter = std::function<std::string(const std::string&, const std::string&, int)>;
  using DiagnosticFormatter = std::function<std::string(const std::string&, const std::string&, int,
                                                        const std::string&, SourceLocation)>;

  Logger(MessageFormatter msgFmt, DiagnosticFormatter diagFmt, std::ostream& os = std::cout,
         bool show = true);

  /// @brief Report message with file, line in dawn source
  MessageProxy operator()(const std::string& filename, int line);

  /// @brief Report message with file, line in dawn source and source,loc in input DSL code.
  DiagnosticProxy operator()(const std::string& file, int line, const std::string& source,
                             SourceLocation loc = SourceLocation());

  /// @brief Add a new message -- called from Proxy objects
  /// {
  void enqueue(std::string msg, const std::string& file, int line);
  void enqueue(std::string msg, const std::string& file, int line, const std::string& source,
               SourceLocation loc);
  /// }

  /// @brief Get and set ostream
  /// {
  std::ostream& stream() const;
  void stream(std::ostream& os);
  /// }

  /// @brief Get and set MessageFormatter
  /// {
  MessageFormatter messageFormatter() const;
  void messageFormatter(const MessageFormatter& fmt);
  /// }

  /// @brief Get and set DiagnosticFormatter
  /// {
  DiagnosticFormatter diagnosticFormatter() const;
  void diagnosticFormatter(const DiagnosticFormatter& fmt);
  /// }

  /// @brief Reset storage
  void clear();

  /// @brief Show or hide output from ostream -- still accessible in the container
  /// {
  void show();
  void hide();
  /// }

  // Expose container of messages
  using iterator = Container::iterator;
  iterator begin();
  iterator end();

  using const_iterator = Container::const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

  Container::size_type size() const;

private:
  void doEnqueue(const std::string& message);

  MessageFormatter msgFmt_;
  DiagnosticFormatter diagFmt_;
  std::ostream* os_;
  Container data_;
  bool show_;
};

/// @brief create a basic (default) message formatter
Logger::MessageFormatter makeMessageFormatter(const std::string type = "");

/// @brief create a basic (default) diagnostic formatter
Logger::DiagnosticFormatter makeDiagnosticFormatter(const std::string type = "");

/// @brief Stack Track object for diagnostics
using DiagnosticStack = std::stack<std::tuple<std::string, SourceLocation>>;

/// @brief Create a stack trace string for diagnostics
std::string createDiagnosticStackTrace(const std::string& prefix, const DiagnosticStack& stack);

namespace log {
// Loggers used for information and warnings
extern Logger info;
extern Logger warn;
extern Logger error;

enum class Level { All, Warnings, Errors, None };

void setVerbosity(Level level);

} // namespace log

} // namespace dawn

/// @macro DAWN_LOG
/// @brief Loggging macros
/// @ingroup support
#define DAWN_LOG(Level) DAWN_LOG_##Level##_IMPL()

#define DAWN_LOG_INFO_IMPL() dawn::log::info(__FILE__, __LINE__)
#define DAWN_LOG_WARNING_IMPL() dawn::log::warn(__FILE__, __LINE__)
#define DAWN_LOG_ERROR_IMPL() dawn::log::error(__FILE__, __LINE__)

/// @macro DAWN_DIAG
/// @brief Loggging macros
/// @ingroup support
#define DAWN_DIAG(Level, file, loc) DAWN_DIAG_##Level##_IMPL(file, loc)

#define DAWN_DIAG_INFO_IMPL(file, loc) dawn::log::info(__FILE__, __LINE__, file, loc)
#define DAWN_DIAG_WARNING_IMPL(file, loc) dawn::log::warn(__FILE__, __LINE__, file, loc)
#define DAWN_DIAG_ERROR_IMPL(file, loc) dawn::log::error(__FILE__, __LINE__, file, loc)
