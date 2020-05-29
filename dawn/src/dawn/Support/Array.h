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

#pragma once

#include "dawn/Support/HashCombine.h"

#include <array>
#include <iosfwd>
#include <ostream>

namespace dawn {

/// @name Common array types
/// @ingroup support
/// @{
using Array2i = std::array<int, 2>;
using Array3i = std::array<int, 3>;
using Array3ui = std::array<unsigned int, 3>;
/// @}

extern std::ostream& operator<<(std::ostream& os, const Array3i& array);
extern std::ostream& operator<<(std::ostream& os, const Array2i& array);

} // namespace dawn

namespace std {

template <>
struct hash<dawn::Array2i> {
  size_t operator()(const dawn::Array2i& array) const {
    size_t seed = 0;
    dawn::hash_combine(seed, array[0], array[1]);
    return seed;
  }
};

template <>
struct hash<dawn::Array3i> {
  size_t operator()(const dawn::Array3i& array) const {
    size_t seed = 0;
    dawn::hash_combine(seed, array[0], array[1], array[2]);
    return seed;
  }
};

} // namespace std
