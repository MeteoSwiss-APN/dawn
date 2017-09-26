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

#include "gsl/Support/SourceLocation.h"
#include <iostream>

namespace gsl {

std::ostream& operator<<(std::ostream& os, const SourceLocation& sourceLocation) {
  return (os << sourceLocation.Line << ":" << sourceLocation.Column);
}

extern bool operator==(const SourceLocation& a, const SourceLocation& b) {
  return a.Line == b.Line && b.Column == b.Column;
}
extern bool operator!=(const SourceLocation& a, const SourceLocation& b) { return !(a == b); }

} // namespace gsl
