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
 * @see https://en.cppreference.com/w/cpp/numeric/math/pow
 */
template <typename T, typename U>
GT_FUNCTION T pow(const T x, const U y) {
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
template <typename T, typename U>
GT_FUNCTION auto min(const T x, const U y) -> decltype(x + y) {
  return x < y ? x : y;
}

/**
 * @brief Returns the greater value of @c x and @c y
 *
 * @see http://en.cppreference.com/w/cpp/algorithm/max
 */
template <typename T, typename U>
GT_FUNCTION auto max(const T x, const U y) -> decltype(x + y) {
  return x > y ? x : y;
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

/**
 * @brief Computes the sine of x
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/sin
 */
template <typename T>
GT_FUNCTION T sin(const T x) {
  return ::sin(x);
}

/**
 * @brief Computes the cosine of x
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/cos
 */
template <typename T>
GT_FUNCTION T cos(const T x) {
  return ::cos(x);
}

/**
 * @brief Computes the tangent of x
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/tan
 */
template <typename T>
GT_FUNCTION T tan(const T x) {
  return ::tan(x);
}

/**
 * @brief Computes the arc sine of x
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/asin
 */
template <typename T>
GT_FUNCTION T asin(const T x) {
  return ::asin(x);
}

/**
 * @brief Computes the arc cosine of x
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/acos
 */
template <typename T>
GT_FUNCTION T acos(const T x) {
  return ::acos(x);
}

/**
 * @brief Computes the arc tangent of x
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/atan
 */
template <typename T>
GT_FUNCTION T atan(const T x) {
  return ::atan(x);
}

/**
 * @brief Determines if the given floating point number arg has finite value i.e. it is normal,
 * subnormal or zero, but not infinite or NaN.
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/isfinite
 */
template <typename T>
GT_FUNCTION T isfinite(const T x) {
  return std::isfinite(x);
}

/**
 * @brief Determines if the given floating point number arg is a positive or negative infinity.
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/isinf
 */
template <typename T>
GT_FUNCTION T isinf(const T x) {
  return std::isinf(x);
}

/**
 * @brief Determines if the given floating point number arg is a not-a-number (NaN) value.
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/isnan
 */
template <typename T>
GT_FUNCTION T isnan(const T x) {
  return std::isnan(x);
}


template <typename T>
GT_FUNCTION T sign(const T val) {
  return (T(0) < val) - (val < T(0));;
}
/** @} */
} // namespace math
} // namespace dawn
} // namespace gridtools
