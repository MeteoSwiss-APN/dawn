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

#ifndef GSL_SUPPORT_MATHEXTRAS_H
#define GSL_SUPPORT_MATHEXTRAS_H

#include <cstdint>

namespace gsl {

/// @brief Returns the next power of two (in 64-bits) that is strictly greater than A. Returns zero
/// on overflow.
///
/// @ingroup support
inline std::uint64_t nextPowerOf2(std::uint64_t A) {
  A |= (A >> 1);
  A |= (A >> 2);
  A |= (A >> 4);
  A |= (A >> 8);
  A |= (A >> 16);
  A |= (A >> 32);
  return A + 1;
}

} // namespace gsl

#endif
