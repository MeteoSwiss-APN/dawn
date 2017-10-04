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

#include "dawn/Support/Logging.h"

#ifndef GTCLANG_SUPPORT_LOGGER_H
#define GTCLANG_SUPPORT_LOGGER_H

namespace gtclang {

/// @brief Logger implementation
/// @ingroup support
class Logger : public dawn::LoggerInterface {
public:
  /// @brief Log `message` of severity `level` at position `file:line`
  virtual void log(dawn::LoggingLevel level, const std::string& message, const char* file,
                   int line) override;
};

} // namespace gtclang

#endif
