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

#include "dawn/Support/Array.h"
#include "dawn/Support/StringUtil.h"

namespace dawn {

std::ostream& operator<<(std::ostream& os, const Array2i& array) {
  return (os << RangeToString()(array));
}

std::ostream& operator<<(std::ostream& os, const Array3i& array) {
  return (os << RangeToString()(array));
}

} // namespace dawn
