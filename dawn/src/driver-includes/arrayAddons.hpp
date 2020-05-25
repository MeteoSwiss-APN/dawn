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

#include <cstddef>
#include <array>

namespace gridtools {
namespace dawn {

template <std::size_t N>
std::array<int, N> operator+(std::array<int, N> const& lhs, std::array<int, N> const& rhs) {
  std::array<int, N> res;
  for(std::size_t i = 0; i < N; ++i) {
    res[i] = lhs[i] + rhs[i];
  }
  return res;
}
} // namespace dawn
} // namespace gridtools
