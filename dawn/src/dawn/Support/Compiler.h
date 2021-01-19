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

#include "dawn/Support/Config.h"

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#ifndef __has_extension
#define __has_extension(x) 0
#endif

#ifndef __has_attribute
#define __has_attribute(x) 0
#endif

#ifndef __has_cpp_attribute
#define __has_cpp_attribute(x) 0
#endif

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if defined(__clang__)
#define DAWN_COMPILER_CLANG 1
#endif

#if defined(__ICC) || defined(__INTEL_COMPILER)
#define DAWN_COMPILER_INTEL 1
#endif

#if defined(__GNUC__) || defined(__GNUG__)
#define DAWN_COMPILER_GNU 1
#endif

#if defined(_MSC_VER)
#define DAWN_COMPILER_MSVC 1
#endif

/// @macro DAWN_GNUC_PREREQ
/// @brief Extend the default `__GNUC_PREREQ` even if glibc's `features.h` isn't available
/// @ingroup support
#ifndef DAWN_GNUC_PREREQ
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && defined(__GNUC_PATCHLEVEL__)
#define DAWN_GNUC_PREREQ(maj, min, patch)                                                          \
  ((__GNUC__ << 20) + (__GNUC_MINOR__ << 10) + __GNUC_PATCHLEVEL__ >=                              \
   ((maj) << 20) + ((min) << 10) + (patch))
#elif defined(__GNUC__) && defined(__GNUC_MINOR__)
#define DAWN_GNUC_PREREQ(maj, min, patch)                                                          \
  ((__GNUC__ << 20) + (__GNUC_MINOR__ << 10) >= ((maj) << 20) + ((min) << 10))
#else
#define DAWN_GNUC_PREREQ(maj, min, patch) 0
#endif
#endif

/// @macro DAWN_BUILTIN_UNREACHABLE
/// @brief Indicate unreachable state
///
/// On compilers which support it, expands to an expression which states that it is undefined
/// behaviour for the compiler to reach this point. Otherwise is not defined.
///
/// @ingroup support
#if __has_builtin(__builtin_unreachable) || DAWN_GNUC_PREREQ(4, 5, 0)
#define DAWN_BUILTIN_UNREACHABLE __builtin_unreachable()
#elif defined(_MSC_VER)
#define DAWN_BUILTIN_UNREACHABLE __assume(false)
#endif

/// @macro DAWN_ATTRIBUTE_ALWAYS_INLINE
/// @brief Mark a method as "always inline" for performance reasons
/// @ingroup support
#if __has_attribute(always_inline) || DAWN_GNUC_PREREQ(4, 0, 0)
#define DAWN_ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline))
#elif defined(_MSC_VER)
#define DAWN_ATTRIBUTE_ALWAYS_INLINE __forceinline
#else
#define DAWN_ATTRIBUTE_ALWAYS_INLINE
#endif

/// @macro DAWN_ATTRIBUTE_NORETURN
/// @brief Mark a method as "no return"
/// @ingroup support
#ifdef __GNUC__
#define DAWN_ATTRIBUTE_NORETURN __attribute__((noreturn))
#elif defined(_MSC_VER)
#define DAWN_ATTRIBUTE_NORETURN __declspec(noreturn)
#else
#define DAWN_ATTRIBUTE_NORETURN
#endif

/// @macro DAWN_BUILTIN_LIKELY
/// @brief Mark this expression as being likely evaluated to "true"
/// @ingroup support
#if __has_builtin(__builtin_expect) || DAWN_GNUC_PREREQ(4, 5, 0)
#define DAWN_BUILTIN_LIKELY(x) __builtin_expect(!!(x), 1)
#else
#define DAWN_BUILTIN_LIKELY(x) (x)
#endif

/// @macro DAWN_BUILTIN_UNLIKELY
/// @brief Mark this expression as being likely evaluated to "false"
/// @ingroup support
#if __has_builtin(__builtin_expect) || DAWN_GNUC_PREREQ(4, 5, 0)
#define DAWN_BUILTIN_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define DAWN_BUILTIN_UNLIKELY(x) (x)
#endif

/// @macro DAWN_BUILTIN_UNREACHABLE
/// @brief Indicate unreachable state
///
/// On compilers which support it, expands to an expression which states that it is undefined
/// behaviour for the compiler to reach this point. Otherwise is not defined.
#if __has_builtin(__builtin_unreachable) || DAWN_GNUC_PREREQ(4, 5, 0)
#define DAWN_BUILTIN_UNREACHABLE __builtin_unreachable()
#elif defined(_MSC_VER)
#define DAWN_BUILTIN_UNREACHABLE __assume(false)
#endif

/// @macro DAWN_ALIGNAS
/// @brief Used to specify a minimum alignment for a structure or variable
/// @ingroup support
#if __GNUC__ && !__has_feature(cxx_alignas) && !DAWN_GNUC_PREREQ(4, 8, 1)
#define DAWN_ALIGNAS(x) __attribute__((aligned(x)))
#else
#define DAWN_ALIGNAS(x) alignas(x)
#endif

/// @macro DAWN_ATTRIBUTE_UNUSED
/// @brief Indicate a function, variable or class is unused
///
/// Some compilers warn about unused functions. When a function is sometimes used or not depending
/// on build settings (e.g. a function only called from  within "assert"), this attribute can be
/// used to suppress such warnings.
///
/// However, it shouldn't be used for unused *variables*, as those have a much more portable
/// solution:
///
/// @code
///   (void)unused_var_name;
/// @endcode
///
/// Prefer cast-to-void wherever it is sufficient.
/// @ingroup support
#if __has_attribute(unused) || DAWN_GNUC_PREREQ(3, 1, 0)
#define DAWN_ATTRIBUTE_UNUSED __attribute__((__unused__))
#else
#define DAWN_ATTRIBUTE_UNUSED
#endif

/// @macro DAWN_NORETURN
/// @brief Indicate a function will never return
/// @ingroup support
#ifdef __GNUC__
#define DAWN_ATTRIBUTE_NORETURN __attribute__((noreturn))
#elif defined(_MSC_VER)
#define DAWN_ATTRIBUTE_NORETURN __declspec(noreturn)
#else
#define DAWN_ATTRIBUTE_NORETURN
#endif

/// @macro DAWN_CURRENT_FUNCTION
/// @brief Name of the current function
/// @ingroup support
#if defined(_MSC_VER)
#define DAWN_CURRENT_FUNCTION __FUNCSIG__
#elif defined(__GNUC__) || (defined(__MWERKS__) && (__MWERKS__ >= 0x3000)) ||                      \
    (defined(__ICC) && (__ICC >= 600)) || defined(__ghs__)
#define DAWN_CURRENT_FUNCTION __PRETTY_FUNCTION__
#elif defined(__DMC__) && (__DMC__ >= 0x810)
#define DAWN_CURRENT_FUNCTION __PRETTY_FUNCTION__
#elif defined(__FUNCSIG__)
#define DAWN_CURRENT_FUNCTION __FUNCSIG__
#elif(defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 600)) ||                                   \
    (defined(__IBMCPP__) && (__IBMCPP__ >= 500))
#define DAWN_CURRENT_FUNCTION __FUNCTION__
#elif defined(__BORLANDC__) && (__BORLANDC__ >= 0x550)
#define DAWN_CURRENT_FUNCTION __FUNC__
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)
#define DAWN_CURRENT_FUNCTION __func__
#elif defined(__cplusplus) && (__cplusplus >= 201103)
#define DAWN_CURRENT_FUNCTION __func__
#else
#define DAWN_CURRENT_FUNCTION "(unknown)"
#endif
