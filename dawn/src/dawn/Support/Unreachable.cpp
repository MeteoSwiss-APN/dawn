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

#include "dawn/Support/Unreachable.h"

#include <cstdio>
#include <iostream>

namespace dawn {

DAWN_ATTRIBUTE_NORETURN void dawn_unreachable_internal(const char* msg, const char* file,
                                                       unsigned line) {
  std::cerr << "FATAL ERROR: UNREACHABLE executed : ";
  if(msg)
    std::cerr << "\"" << msg << "\"";
  if(file)
    std::cerr << " at " << file << ":" << line;
  std::cerr << std::endl;
  std::abort();

#ifdef DAWN_BUILTIN_UNREACHABLE
  DAWN_BUILTIN_UNREACHABLE;
#endif
}

} // namespace dawn
