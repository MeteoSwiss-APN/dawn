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

namespace gtclang {

namespace dsl {

/**
 * @namespace math
 * @brief Namespace of DSL's math functions
 *
 * @see math_function
 */
namespace math {

/**
 * @defgroup math_function Math functions
 * @brief Math functions which can be safely used inside stencil and stencil_function
 * Do-methods
 *
 * @ingroup gtclang_dsl
 * @{
 */

/**
 * @brief Computes the absolute value of a floating point value @c arg
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/fabs
 */
template <typename T>
T fabs(const T x);

/**
 * @brief Computes the largest integer value not greater than @c arg
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/floor
 */
template <typename T>
T floor(const T arg);

/**
 * @brief Computes the smallest integer value not less than @c arg
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/ceil
 */
template <typename T>
T ceil(const T arg);

/**
 * @brief Truncate @c arg to an integer
 *
 * The truncation is performed by casting @c arg to @c int
 */
template <typename T>
int trunc(const T arg);

/**
 * @brief Returns the floating-point remainder of @c x/y (rounded towards zero)
 *
 * @param x Value of the quotient numerator
 * @param y Value of the quotient denominator
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/fmod
 */
template <typename T>
T fmod(const T x, const T y);

/**
 * @brief Raise @c x to power @c y.
 *
 * @param x Base value
 * @param y Exponent value
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/fmod
 */
template <typename T>
T pow(const T x, const T y);

/**
 * @brief Compute square root
 *
 * @param x Value whose square root is computed. If the argument is negative, a domain
 * error occurs.
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/sqrt
 */
template <typename T>
T sqrt(const T x);

/**
 * @brief Returns the smaller value of @c x and @c y
 *
 * @see http://en.cppreference.com/w/cpp/algorithm/min
 */
template <typename T>
T min(const T x, const T y);

/**
 * @brief Returns the greater value of @c x and @c y
 *
 * @see http://en.cppreference.com/w/cpp/algorithm/max
 */
template <typename T>
T max(const T x, const T y);

/**
 * @brief Computes the @c e (Euler's number, 2.7182818) raised to the given power @c arg
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/exp
 */
template <typename T>
T exp(const T arg);

/**
 * @brief Computes the the natural (base @c e) logarithm of arg.
 *
 * @see http://en.cppreference.com/w/cpp/numeric/math/log
 */
template <typename T>
T log(const T x);

/** @} */
} // namespace math
} // namespace dsl
} // namespace gtclang
