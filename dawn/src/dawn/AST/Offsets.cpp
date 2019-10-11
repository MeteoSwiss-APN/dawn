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

#ifndef DAWN_AST_OFFSETS_CPP
#define DAWN_AST_OFFSETS_CPP

#include "Offsets.h"

#include <iostream>

namespace dawn::ast {

std::ostream& operator<<(std::ostream& os, Offsets const& offset) {
  auto const& hoffset = ast::offset_cast<StructuredOffset const&>(offset.horizontalOffset());
  auto const& voffset = offset.verticalOffset();
  return os << hoffset.offsetI() << ", " << hoffset.offsetJ() << ", " << voffset;
}

} // namespace dawn::ast

#endif
