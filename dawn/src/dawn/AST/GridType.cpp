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

#include "dawn/AST/GridType.h"
#include <ostream>

namespace dawn {

std::ostream& operator<<(std::ostream& os, const ast::GridType& gridType) {
  switch(gridType) {
  case ast::GridType::Cartesian:
    os << "structured";
    break;
  case ast::GridType::Unstructured:
    os << "unstructured";
    break;
  }
  return os;
}

} // namespace dawn
