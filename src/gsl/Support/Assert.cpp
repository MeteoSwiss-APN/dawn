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

#include "gsl/Support/Assert.h"
#include <cstdlib>
#include <iostream>

namespace gsl {

GSL_ATTRIBUTE_NORETURN static void assertionFailedImpl(char const* expr, char const* msg,
                                                       char const* function, char const* file,
                                                       long line) {

  std::cerr << "Assertion failed: `" << expr << "' " << (msg == nullptr ? "" : msg) << "\n"
            << "Function: '" << function << "'\n"
            << "Location: " << file << ":" << line << std::endl;
  std::abort();
  // std::exit(EXIT_FAILURE);
}

void assertionFailed(char const* expr, char const* function, char const* file, long line) {
  assertionFailedImpl(expr, nullptr, function, file, line);
}

void assertionFailedMsg(char const* expr, char const* msg, char const* function, char const* file,
                        long line) {
  assertionFailedImpl(expr, msg, function, file, line);
}

} // namespace gsl
