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

/**
 * @name Storage defintions
 * @ingroup dawn
 * @{
 */
#define DAWN_STORAGE_DUMMY 0
#define DAWN_STORAGE_HOST 1
#define DAWN_STORAGE_CUDA 2
/** @} */

/**
 * @name Grid defintions
 * @ingroup dawn
 * @{
 */
#define DAWN_GRID_STRUCTURED 0
/** @} */

/**
 * @name Floating point precisions
 * @ingroup dawn
 * @{
 */
#define DAWN_SINGLE_PRECISION 0
#define DAWN_DOUBLE_PRECISION 1
/** @} */

// Default precision:
//  1. Honor defintions of gridtools GT_FLOAT_PRECISION
//  2. Fall back to double precision
#ifndef DAWN_PRECISION

#ifdef GT_FLOAT_PRECISION

#if GT_FLOAT_PRECISION == 4
#define DAWN_PRECISION DAWN_SINGLE_PRECISION
#else
#define DAWN_PRECISION DAWN_DOUBLE_PRECISION
#endif

#else // !defined(GT_FLOAT_PRECISION)

#define DAWN_PRECISION DAWN_DOUBLE_PRECISION

#endif // GT_FLOAT_PRECISION

#endif // DAWN_PRECISION

namespace dawn {

/**
 * @typedef float_type
 * @brief Floating point type [default: `double`]
 *
 * This is either a `double` or `float` depending if `DAWN_SINGLE_PRECISION` or
 * `DAWN_DOUBLE_PRECISION` is defined. Note that it is strongly adviced to use this
 * typedef instead of `double` or `float` directly as the latter may incur unnecessary casts
 * in CUDA code.
 *
 * @ingroup dawn
 */
#if DAWN_PRECISION == DAWN_SINGLE_PRECISION
using float_type = float;
#elif DAWN_PRECISION == DAWN_DOUBLE_PRECISION
using float_type = double;
#else
#error DAWN_PRECISION is invalid
#endif
} // namespace dawn
