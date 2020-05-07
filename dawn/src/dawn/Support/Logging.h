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

#include <functional>
#include <iostream>
#include <list>
#include <sstream>
#include <string>

namespace dawn {

enum class LoggingLevel { Info, Warning, Error, Fatal };

class Logger;

class LoggerProxy {
public:
  LoggerProxy(Logger& logger, const std::string& source, int line);
  LoggerProxy(const LoggerProxy& other);
  ~LoggerProxy();

  template <typename Streamable>
  LoggerProxy& operator<<(Streamable&& obj) {
    ss_ << obj;
    return *this;
  }

private:
  Logger& logger_;
  std::stringstream ss_;
  const std::string source_;
  const int line_;
};

class Logger {
public:
  using MessageContainer = std::list<std::string>;
  using Formatter = std::function<std::string(const std::string&, const std::string&, int)>;

  Logger(Formatter fmt, std::ostream& os = std::cout);

  LoggerProxy operator()(const std::string& source, int line);

  void enqueue(const std::string& msg, const std::string& file, int line);

  // Get and set ostream
  std::ostream& stream() const;
  void stream(std::ostream& os);

  // Get and set Formatter
  Formatter formatter() const;
  void formatter(const Formatter& fmt);

  // Reset storage
  void clear();

  // Expose container of messages
  using iterator = MessageContainer::iterator;
  iterator begin();
  iterator end();

  using const_iterator = MessageContainer::const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

  MessageContainer::size_type size() const;

private:
  Formatter fmt_;
  std::ostream* os_;
  MessageContainer data_;
};

Logger::Formatter makeDefaultFormatter(const std::string prefix);

extern Logger info;
extern Logger warn;
extern Logger error;

} // namespace dawn

/// @macro DAWN_LOG
/// @brief Loggging macros
/// @ingroup support
#define DAWN_LOG(Level) DAWN_LOG_##Level##_IMPL()

#define DAWN_LOG_INFO_IMPL() dawn::info(__FILE__, __LINE__)
#define DAWN_LOG_WARNING_IMPL() dawn::warn(__FILE__, __LINE__)
#define DAWN_LOG_ERROR_IMPL() dawn::error(__FILE__, __LINE__)

#endif
