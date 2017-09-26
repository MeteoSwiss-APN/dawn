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

#ifndef GSL_UNITTEST_UNITTESTLOGGER_H
#define GSL_UNITTEST_UNITTESTLOGGER_H

#include "gsl/Support/Logging.h"

namespace gsl {

/// @brief Simple logger to std::cout for debugging purposes
class UnittestLogger : public LoggerInterface {
public:
  void log(LoggingLevel level, const std::string& message, const char* file, int line) override;
};

} // namespace gsl

#endif
