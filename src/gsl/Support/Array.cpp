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

#include "gsl/Support/Array.h"
#include "gsl/Support/StringUtil.h"
#include <iostream>

namespace gsl {

std::ostream& operator<<(std::ostream& os, const Array2i& array) {
  return (os << RangeToString()(array));
}

std::ostream& operator<<(std::ostream& os, const Array3i& array) {
  return (os << RangeToString()(array));
}

} // namespace gsl
