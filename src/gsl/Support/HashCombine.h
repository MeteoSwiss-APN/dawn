//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#ifndef GSL_SUPPORT_HASHCOMBINE_H
#define GSL_SUPPORT_HASHCOMBINE_H

#include <functional>

namespace gsl {

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
///   gsl::hash_combine(hash, a, b);
/// @endcode
///
/// @see boost::hash_combine
/// @ingroup support
/// @{
inline void hash_combine(std::size_t& seed) {}

template <typename T, typename... Rest>
inline void hash_combine(std::size_t& seed, const T& v, Rest... rest) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  hash_combine(seed, rest...);
}
/// @}

} // namespace gsl

#endif
