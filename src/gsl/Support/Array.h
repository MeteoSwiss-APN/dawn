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

#ifndef GSL_SUPPORT_ARRAY_H
#define GSL_SUPPORT_ARRAY_H

#include "gsl/Support/HashCombine.h"
#include <array>
#include <iosfwd>

namespace gsl {

/// @name Common array types
/// @ingroup support
/// @{
using Array2i = std::array<int, 2>;
using Array3i = std::array<int, 3>;
/// @}

extern std::ostream& operator<<(std::ostream& os, const Array3i& array);
extern std::ostream& operator<<(std::ostream& os, const Array2i& array);

} // namespace gsl

namespace std {

template <>
struct hash<gsl::Array2i> {
  size_t operator()(const gsl::Array2i& array) const {
    size_t hash = 0;
    gsl::hash_combine(hash, array[0], array[1]);
    return hash;
  }
};

template <>
struct hash<gsl::Array3i> {
  size_t operator()(const gsl::Array3i& array) const {
    size_t hash = 0;
    gsl::hash_combine(hash, array[0], array[1], array[2]);
    return hash;
  }
};

} // namespace std

#endif
