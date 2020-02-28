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

#include <array>

namespace dawn {
namespace driver {

// Access dimensions with `auto dim_extent = get<D>();`, where D = 0,1,2 (=i,j,k)
// and minus/plus component with `get<C>(dim_extent)`, where C = 0 (minus), 1 (plus)
// (Note, need `using std::get` for ADL to kick-in if std < c++20)
using cartesian_extent = std::array<std::array<int, 2>, 3>;

// Access dimensions with get<D>(), where D = 0,1 (=horizontal, vertical)
// for D == 0 it is either true (have extent) or false (no extent)
// for D == 1, see Cartesian case.
struct unstructured_extent {
  bool horizontal;
  std::array<int, 2> vertical;
};

namespace detail {
template <int I>
struct get_impl;

template <>
struct get_impl<0> {
  using type = bool;
  constexpr type operator()(unstructured_extent const& extent) { return extent.horizontal; }
};
template <>
struct get_impl<1> {
  using type = std::array<int, 2>;
  constexpr type operator()(unstructured_extent const& extent) { return extent.vertical; }
};

} // namespace detail

template <int I>
constexpr auto get(unstructured_extent const& extent) -> decltype(detail::get_impl<I>{}(extent)) {
  return detail::get_impl<I>{}(extent);
}

} // namespace driver
} // namespace dawn
