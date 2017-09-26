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

#include "gsl/Support/Unreachable.h"
#include <cstdio>
#include <iostream>

namespace gsl {

GSL_ATTRIBUTE_NORETURN void gsl_unreachable_internal(const char* msg, const char* file,
                                                     unsigned line) {
  std::cerr << "FATAL ERROR: UNREACHABLE executed : ";
  if(msg)
    std::cerr << "\"" << msg << "\"";
  if(file)
    std::cerr << " at " << file << ":" << line;
  std::cerr << std::endl;
  std::abort();

#ifdef GSL_BUILTIN_UNREACHABLE
  GSL_BUILTIN_UNREACHABLE;
#endif
}

} // namespace gsl
