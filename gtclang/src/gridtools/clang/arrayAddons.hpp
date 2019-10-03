//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _       _
//                        | |     | |
//                    __ _| |_ ___| | __ _ _ __   __ _
//                   / _` | __/ __| |/ _` | '_ \ / _` |
//                  | (_| | || (__| | (_| | | | | (_| |
//                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
//                    __/ |                       __/ |
//                   |___/                       |___/
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#ifndef GRIDTOOLS_CLANG_ARRAYADDONS_HPP
#define GRIDTOOLS_CLANG_ARRAYADDONS_HPP

#include <array>

namespace gridtools {
namespace clang {

template <size_t N>
std::array<int, N> operator+(std::array<int, N> const& lhs, std::array<int, N> const& rhs) {
  std::array<int, N> res;
  for(size_t i = 0; i < N; ++i) {
    res[i] = lhs[i] + rhs[i];
  }
  return res;
}
} // namespace clang
} // namespace gridtools

#endif
