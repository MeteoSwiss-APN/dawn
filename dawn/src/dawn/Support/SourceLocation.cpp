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

#include "dawn/Support/SourceLocation.h"
#include <iostream>

namespace dawn {

std::ostream& operator<<(std::ostream& os, const SourceLocation& sourceLocation) {
  return (os << sourceLocation.Line << ":" << sourceLocation.Column);
}

extern bool operator==(const SourceLocation& a, const SourceLocation& b) {
  return a.Line == b.Line && a.Column == b.Column;
}
extern bool operator!=(const SourceLocation& a, const SourceLocation& b) { return !(a == b); }

} // namespace dawn
