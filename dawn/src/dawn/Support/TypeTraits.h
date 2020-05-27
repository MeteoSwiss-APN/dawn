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

#include <type_traits>
#include <utility>

namespace dawn {

/// @brief This is a type trait that is used to determine whether a given type can be copied around
/// with memcpy instead of running ctors etc.
/// @ingroup support
/// @{
template <typename T>
struct isPodLike {

// std::is_trivially_copyable is available in libc++ with clang, libstdc++
// that comes with GCC 5.
#if(__has_feature(is_trivially_copyable) && defined(_LIBCPP_VERSION)) ||                           \
    (defined(__GNUC__) && __GNUC__ >= 5)
  // If the compiler supports the is_trivially_copyable trait use it, as it
  // matches the definition of isPodLike closely.
  static const bool value = std::is_trivially_copyable<T>::value;
#elif __has_feature(is_trivially_copyable)
  // Use the internal name if the compiler supports is_trivially_copyable but we
  // don't know if the standard library does. This is the case for clang in
  // conjunction with libstdc++ from GCC 4.x.
  static const bool value = __is_trivially_copyable(T);
#else
  // If we don't know anything else, we can (at least) assume that all non-class
  // types are PODs.
  static const bool value = !std::is_class<T>::value;
#endif
};

// std::pair's are pod-like if their elements are.
template <typename T, typename U>
struct isPodLike<std::pair<T, U>> {
  static const bool value = isPodLike<T>::value && isPodLike<U>::value;
};
/// @}

/// @brief If `T` is a pointer, just return it. If it is not, return `T&`
/// @ingroup support
/// @{
template <typename T, typename Enable = void>
struct add_lvalue_reference_if_not_pointer {
  typedef T& type;
};

template <typename T>
struct add_lvalue_reference_if_not_pointer<
    T, typename std::enable_if<std::is_pointer<T>::value>::type> {
  typedef T type;
};
/// @}

/// @brief If `T` is a pointer to `X`, return a pointer to const `X`. If it is not, return const `T`
/// @ingroup support
/// @{
template <typename T, typename Enable = void>
struct add_const_past_pointer {
  typedef const T type;
};

template <typename T>
struct add_const_past_pointer<T, typename std::enable_if<std::is_pointer<T>::value>::type> {
  typedef const typename std::remove_pointer<T>::type* type;
};
/// @}

/// @brief Check if `Cond` evaluates to `true` for a variadic pack
/// /// @ingroup support
/// @{
template <typename... Conds>
struct and_ : std::true_type {};

template <typename Cond, typename... Conds>
struct and_<Cond, Conds...> : std::conditional<Cond::value, and_<Conds...>, std::false_type>::type {
};
/// @}

} // namespace dawn
