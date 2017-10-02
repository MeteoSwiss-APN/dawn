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

#ifndef DAWN_UNITTEST_UNITTESTLOGGER_H
#define DAWN_UNITTEST_UNITTESTLOGGER_H

#include "dawn/Support/Logging.h"

namespace dawn {

/// @brief Simple logger to std::cout for debugging purposes
class UnittestLogger : public LoggerInterface {
public:
  void log(LoggingLevel level, const std::string& message, const char* file, int line) override;
};

} // namespace dawn

#endif
