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

#ifndef DAWN_SUPPORT_LOGGING_H
#define DAWN_SUPPORT_LOGGING_H

#include <memory>
#include <sstream>
#include <string>

namespace dawn {

/// @enum LoggingLevel
/// @brief Severity levels
/// @ingroup support
enum class LoggingLevel { Info = 0, Warning, Error, Fatal };

/// @brief Logging interface
/// @ingroup support
class LoggerInterface {
public:
  virtual ~LoggerInterface() {}

  /// @brief Log `message` of severity `level` at position `file:line`
  ///
  /// @param level     Severity level
  /// @param message   Message to log
  /// @param file      File from which the logging was issued
  /// @param line      Line in `file` from which the logging was issued
  virtual void log(LoggingLevel level, const std::string& message, const char* file, int line) = 0;
};

namespace internal {

class LoggerProxy {
  LoggingLevel level_;
  std::reference_wrapper<std::stringstream> ss_;
  const char* file_;
  int line_;

public:
  LoggerProxy(const LoggerProxy&) = default;
  LoggerProxy(LoggingLevel level, std::stringstream& ss, const char* file, int line);

  ~LoggerProxy();

  template <class StreamableValueType>
  LoggerProxy& operator<<(StreamableValueType&& value) {
    ss_.get() << value;
    return *this;
  }
};

} // namespace internal

/// @brief Dawn Logger adapter
///
/// This is the interface for logging from Dawn. A default logger is provided, but if you would like
/// to register your own, you need to implement the `LoggerInterface`.
///
/// The `DefaultLogger` is a good minimal example.
///
/// @ingroup support
class Logger {
  static Logger* instance_;
  LoggerInterface* logger_;
  std::stringstream ss_;

public:
  /// @brief Initialize Logger object
  Logger();
  ~Logger();

  /// @brief Register a Logger (this does @b not take ownership of the object)
  void registerLogger(LoggerInterface* logger);

  /// @brief Get the current logger or NULL if no logger is currently registered
  LoggerInterface* getLogger();

  /// @name Start logging
  /// @{
  internal::LoggerProxy logInfo(const char* file, int line);
  internal::LoggerProxy logWarning(const char* file, int line);
  internal::LoggerProxy logError(const char* file, int line);
  internal::LoggerProxy logFatal(const char* file, int line);
  /// @}

  /// @brief Log `message` of severity `level` at position `file:line`
  void log(LoggingLevel level, const std::string& message, const char* file, int line);

  /// @brief Get singleton instance
  static Logger& getSingleton();
};

/// @brief Simple logger
/// @ingroup support
class DefaultLogger : public LoggerInterface {
public:
  void log(LoggingLevel level, const std::string& message, const char* file, int line) override;
};

/// @macro DAWN_LOG
/// @brief Loggging macros
/// @ingroup support
#define DAWN_LOG(Level) DAWN_LOG_##Level##_IMPL()

#define DAWN_LOG_INFO_IMPL() dawn::Logger::getSingleton().logInfo(__FILE__, __LINE__)
#define DAWN_LOG_WARNING_IMPL() dawn::Logger::getSingleton().logWarning(__FILE__, __LINE__)
#define DAWN_LOG_ERROR_IMPL() dawn::Logger::getSingleton().logError(__FILE__, __LINE__)
#define DAWN_LOG_FATAL_IMPL() dawn::Logger::getSingleton().logFatal(__FILE__, __LINE__)

} // namespace dawn

#endif
