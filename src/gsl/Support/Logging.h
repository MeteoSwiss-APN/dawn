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

#ifndef GSL_SUPPORT_LOGGING_H
#define GSL_SUPPORT_LOGGING_H

#include <functional>
#include <sstream>
#include <string>

namespace gsl {

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

/// @brief GSL Logger adapter
///
/// To make use of this Logger class, you are required to register your own implementation of a
/// Logger via `registerLogger`. The registered Logger has to implement the `LoggerInterface`.
/// By default no Logger is registered and no logging is performed.
///
/// The following snippet can be seen as a minimal working example:
///
/// @code
///   #include <gsl/Support/Logging.h>
///   #include <iostream>
///
///   class MyLogger : gsl::LoggerInterface {
///   public:
///      void log(LoggingLevel level, const std::string& message, const char* file, int line) {
///        std::cout << file << ":" << line << " " << message << std::endl;
///   };
///
///   int main() {
///     myLogger = new MyLogger;
///     gsl::Logger::getSingleton().registerLogger(myLogger);
///
///     GSL_LOG(INFO) << "Hello world!";
///
///     delete myLogger;
///   }
/// @endcode
///
/// @ingroup support
class Logger {
  static Logger* instance_;
  LoggerInterface* logger_;
  std::stringstream ss_;

public:
  /// @brief Initialize Logger object
  Logger();

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

/// @macro GSL_LOG
/// @brief Loggging macros
/// @ingroup support
#define GSL_LOG(Level) GSL_LOG_##Level##_IMPL()

#define GSL_LOG_INFO_IMPL() gsl::Logger::getSingleton().logInfo(__FILE__, __LINE__)
#define GSL_LOG_WARNING_IMPL() gsl::Logger::getSingleton().logWarning(__FILE__, __LINE__)
#define GSL_LOG_ERROR_IMPL() gsl::Logger::getSingleton().logError(__FILE__, __LINE__)
#define GSL_LOG_FATAL_IMPL() gsl::Logger::getSingleton().logFatal(__FILE__, __LINE__)

} // namespace gsl

#endif
