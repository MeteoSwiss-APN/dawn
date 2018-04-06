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

#pragma once

/**
 * @name Storage defintions
 * @ingroup gridtools_clang
 * @{
 */
#define GRIDTOOLS_CLANG_STORAGE_DUMMY 0
#define GRIDTOOLS_CLANG_STORAGE_HOST 1
#define GRIDTOOLS_CLANG_STORAGE_CUDA 2
/** @} */

/**
 * @name Grid defintions
 * @ingroup gridtools_clang
 * @{
 */
#define GRIDTOOLS_CLANG_GRID_STRUCTURED 0
/** @} */

/**
 * @name Floating point precisions
 * @ingroup gridtools_clang
 * @{
 */
#define GRIDTOOLS_CLANG_SINGLE_PRECISION 0
#define GRIDTOOLS_CLANG_DOUBLE_PRECISION 1
/** @} */

// Default precision:
//  1. Honor defintions of gridtools FLOAT_PRECISION
//  2. Fall back to double precision
#ifndef GRIDTOOLS_CLANG_PRECISION

#ifdef FLOAT_PRECISION

#if FLOAT_PRECISION == 4
#define GRIDTOOLS_CLANG_PRECISION GRIDTOOLS_CLANG_FLOAT_PRECISION
#else
#define GRIDTOOLS_CLANG_PRECISION GRIDTOOLS_CLANG_DOUBLE_PRECISION
#endif
#undef FLOAT_PRECISION

#else // !defined(FLOAT_PRECISION)

#define GRIDTOOLS_CLANG_PRECISION GRIDTOOLS_CLANG_DOUBLE_PRECISION

#endif // FLOAT_PRECISION

#endif // GRIDTOOLS_CLANG_PRECISION

namespace gridtools {

namespace clang {

/**
 * @typedef float_type
 * @brief Floating point type [default: `double`]
 *
 * This is either a `double` or `float` depending if `GRIDTOOLS_CLANG_SINGLE_PRECISION` or
 * `GRIDTOOLS_CLANG_DOUBLE_PRECISION` is defined. Note that it is strongly adviced to use this
 * typedef instead of `double` or `float` directly as the latter may incur unnecessary casts
 * in CUDA code.
 *
 * @ingroup gridtools_clang
 */
#if GRIDTOOLS_CLANG_PRECISION == GRIDTOOLS_CLANG_SINGLE_PRECISION
#define FLOAT_PRECISION 4
using float_type = float;
#elif GRIDTOOLS_CLANG_PRECISION == GRIDTOOLS_CLANG_DOUBLE_PRECISION
#define FLOAT_PRECISION 8
using float_type = double;
#else
#error GRIDTOOLS_CLANG_PRECISION is invalid
#endif
}
}
