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

#include "defs.hpp"
#include "storage.hpp"
#include <algorithm>
#include <cmath>

#ifndef GT_FUNCTION
#define GT_FUNCTION
#endif

namespace gridtools {

namespace dawn {

/**
 * @namespace math
 * @brief Namepsace of gridtools portable math functions
 *
 * @see math_function
 */
namespace math {

/**
 * @defgroup math_function Math functions
 * @brief Math functions implementations
 *
 * @ingroup gridtools_dawn
 * @{
 */

/**
 * @brief Computes the absolute value of a floating point value @c arg
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/fabs
 */
template <typename T>
GT_FUNCTION T fabs(const T x) {
  return ::fabs(x);
}

/**
 * @brief Computes the largest integer value not greater than @c arg
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/floor
 */
template <typename T>
GT_FUNCTION T floor(const T arg) {
  return ::floor(arg);
}

/**
 * @brief Computes the smallest integer value not less than @c arg
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/ceil
 */
template <typename T>
GT_FUNCTION T ceil(const T arg) {
  return ::ceil(arg);
}

/**
 * @brief Truncate @c arg to an integer
 *
 * The truncation is performed by casting @c arg to @c int
 */
template <typename T>
GT_FUNCTION int trunc(const T arg) {
  return static_cast<int>(arg);
}

/**
 * @brief Returns the floating-point remainder of @c x/y (rounded towards zero)
 *
 * @param x Value of the quotient numerator
 * @param y Value of the quotient denominator
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/fmod
 */
template <typename T>
GT_FUNCTION T fmod(const T x, const T y) {
  return ::fmod(x, y);
}

/**
 * @brief Raise @c x to power @c y.
 *
 * @param x Base value
 * @param y Exponent value
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/fmod
 */
template <typename T>
GT_FUNCTION T pow(const T x, const T y) {
  return ::pow(x, y);
}

/**
 * @brief Compute square root
 *
 * @param x Value whose square root is computed. If the argument is negative, a domain
 * error occurs.
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/sqrt
 */
template <typename T>
GT_FUNCTION T sqrt(const T x) {
  return ::sqrt(x);
}

/**
 * @brief Returns the smaller value of @c x and @c y
 *
 * @see http://en.cppreference.com/w/cpp/algorithm/min
 */
template <typename T>
GT_FUNCTION T min(const T x, const T y) {
#if DAWN_STORAGE_TYPE == DAWN_STORAGE_CUDA
  return x < y ? x : y;
#else
  return std::min(x, y);
#endif
}

/**
 * @brief Returns the greater value of @c x and @c y
 *
 * @see http://en.cppreference.com/w/cpp/algorithm/max
 */
template <typename T>
GT_FUNCTION T max(const T x, const T y) {
#if DAWN_STORAGE_TYPE == DAWN_STORAGE_CUDA
  return x > y ? x : y;
#else
  return std::max(x, y);
#endif
}

/**
 * @brief Computes the @c e (Euler's number, 2.7182818) raised to the given power @c arg
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/exp
 */
template <typename T>
GT_FUNCTION T exp(const T arg) {
  return ::exp(arg);
}

/**
 * @brief Computes the the natural (base @c e) logarithm of arg.
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/log
 */
template <typename T>
GT_FUNCTION T log(const T x) {
  return ::log(x);
}

/** @} */
} // namespace math
} // namespace dawn
} // namespace gridtools
