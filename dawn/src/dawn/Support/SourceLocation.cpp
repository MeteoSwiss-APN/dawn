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

#include <string>

namespace dawn {

SourceLocation::operator std::string() const {
  return std::to_string(Line) + ":" + std::to_string(Column);
}

std::ostream& operator<<(std::ostream& os, const SourceLocation& sourceLocation) {
  return os << static_cast<std::string>(sourceLocation);
}

bool operator==(const SourceLocation& a, const SourceLocation& b) {
  return a.Line == b.Line && a.Column == b.Column;
}
bool operator!=(const SourceLocation& a, const SourceLocation& b) { return !(a == b); }

} // namespace dawn
