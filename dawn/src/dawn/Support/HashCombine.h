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

#include <functional>

#include "dawn/AST/LocationType.h"

namespace dawn {

/// @brief Called repeatedly to incrementally create a hash value from several variables
///
/// Note that the order of arguments matters, meaning:
/// `hash_combine(arg1, arg2) != hash_combine(arg2, arg1)`
///
/// @code
///   int a = 5;
///   int b = 2;
///
///   std::size_t hash = 0; // Hash of a and b
///   dawn::hash_combine(hash, a, b);
/// @endcode
///
/// @see boost::hash_combine
/// @ingroup support
/// @{
///

inline void hash_combine(std::size_t& seed) {}

template <typename T, typename... Rest>
inline void hash_combine(std::size_t& seed, const T& v, Rest... rest) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  hash_combine(seed, rest...);
}

/// @}
} // namespace dawn

//  from: https://gist.github.com/angeleno/e838a35f0849ecab56e8be7e46645177
//  https://stackoverflow.com/a/6894436/916549

template <class... TupleArgs>
struct std::hash<std::tuple<TupleArgs...>> {
private:
  //  this is a termination condition
  //  N == sizeof...(TupleTypes)
  //
  template <size_t Idx, typename... TupleTypes>
  inline typename std::enable_if<Idx == sizeof...(TupleTypes), void>::type
  hash_combine_tup(size_t& seed, const std::tuple<TupleTypes...>& tup) const {}

  //  this is the computation function
  //  continues till condition N < sizeof...(TupleTypes) holds
  //
  template <size_t Idx, typename... TupleTypes>
      inline typename std::enable_if <
      Idx<sizeof...(TupleTypes), void>::type
      hash_combine_tup(size_t& seed, const std::tuple<TupleTypes...>& tup) const {
    dawn::hash_combine(seed, std::get<Idx>(tup));

    //  on to next element
    hash_combine_tup<Idx + 1>(seed, tup);
  }

public:
  size_t operator()(const std::tuple<TupleArgs...>& tupleValue) const {
    size_t seed = 0;
    //  begin with the first iteration
    hash_combine_tup<0>(seed, tupleValue);
    return seed;
  }
};
