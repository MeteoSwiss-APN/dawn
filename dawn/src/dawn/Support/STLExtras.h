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

#include "dawn/Support/Compiler.h"
#include <algorithm>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace dawn {

/// @ingroup support
/// @{

//===------------------------------------------------------------------------------------------===//
//     Extra additions to <functional>
//===------------------------------------------------------------------------------------------===//

/// @brief An efficient, type-erasing, non-owning reference to a callable
///
/// This is intended for use as the type of a function parameter that is not used after the function
/// in question returns. This class does not own the callable, so it is not in general safe to store
/// a function_ref.
template <typename Fn>
class function_ref;

template <typename Ret, typename... Params>
class function_ref<Ret(Params...)> {
  Ret (*callback)(intptr_t callable_, Params... params);
  intptr_t callable_;

  template <typename Callable>
  static Ret callback_fn(intptr_t callable, Params... params) {
    return (*reinterpret_cast<Callable*>(callable))(std::forward<Params>(params)...);
  }

public:
  template <typename Callable>
  function_ref(Callable&& callable,
               typename std::enable_if<!std::is_same<typename std::remove_reference<Callable>::type,
                                                     function_ref>::value>::type* = nullptr)
      : callback(callback_fn<typename std::remove_reference<Callable>::type>),
        callable_(reinterpret_cast<intptr_t>(&callable)) {}
  Ret operator()(Params... params) const {
    return callback(callable_, std::forward<Params>(params)...);
  }
};

//===------------------------------------------------------------------------------------------===//
//     Extra additions to <array>
//===------------------------------------------------------------------------------------------===//

/// @brief Get length of an array
template <class T, std::size_t N>
constexpr inline size_t array_lengthof(T (&)[N]) {
  return N;
}

//===------------------------------------------------------------------------------------------===//
//     Extra additions to <algorithm>
//===------------------------------------------------------------------------------------------===//

/// @brief  For a container of pointers, deletes the pointers and then clears the container.
template <typename Container>
void deleteContainerPointers(Container& C) {
  for(auto V : C)
    delete V;
  C.clear();
}

/// @brief  In a container of pairs (usually a map) whose second element is a pointer, deletes the
/// second elements and then clears the container.
template <typename Container>
void deleteContainerSeconds(Container& C) {
  for(auto& V : C)
    delete V.second;
  C.clear();
}

/// @brief  Provide wrappers to std::all_of which take ranges instead of having to pass begin/end
/// explicitly
template <typename R, typename UnaryPredicate>
bool all_of(R&& Range, UnaryPredicate P) {
  return std::all_of(std::begin(Range), std::end(Range), P);
}

/// @brief  Provide wrappers to std::any_of which take ranges instead of having to pass begin/end
/// explicitly
template <typename R, typename UnaryPredicate>
bool any_of(R&& Range, UnaryPredicate P) {
  return std::any_of(std::begin(Range), std::end(Range), P);
}

/// @brief  Provide wrappers to std::none_of which take ranges instead of having to pass begin/end
/// explicitly
template <typename R, typename UnaryPredicate>
bool none_of(R&& Range, UnaryPredicate P) {
  return std::none_of(std::begin(Range), std::end(Range), P);
}

/// @brief  Provide wrappers to std::find which take ranges instead of having to pass begin/end
/// explicitly.
template <typename R, typename T>
auto find(R&& Range, const T& Val) -> decltype(std::begin(Range)) {
  return std::find(std::begin(Range), std::end(Range), Val);
}

/// @brief Provide wrappers to std::find_if which take ranges instead of having to pass begin/end
/// explicitly
template <typename R, typename UnaryPredicate>
auto find_if(R&& Range, UnaryPredicate P) -> decltype(std::begin(Range)) {
  return std::find_if(std::begin(Range), std::end(Range), P);
}

template <typename R, typename UnaryPredicate>
auto find_if_not(R&& Range, UnaryPredicate P) -> decltype(std::begin(Range)) {
  return std::find_if_not(std::begin(Range), std::end(Range), P);
}

/// @brief  Provide wrappers to std::remove_if which take ranges instead of having to pass begin/end
/// explicitly
template <typename R, typename UnaryPredicate>
auto remove_if(R&& Range, UnaryPredicate P) -> decltype(std::begin(Range)) {
  return std::remove_if(std::begin(Range), std::end(Range), P);
}

/// @brief  Wrapper function around std::find to detect if an element exists in a container
template <typename R, typename E>
bool is_contained(R&& Range, const E& Element) {
  return std::find(std::begin(Range), std::end(Range), Element) != std::end(Range);
}

/// @brief  Wrapper function around std::count to count the number of times an element @p Element
/// occurs in the given range \p Range
template <typename R, typename E>
auto count(R&& Range, const E& Element) ->
    typename std::iterator_traits<decltype(std::begin(Range))>::difference_type {
  return std::count(std::begin(Range), std::end(Range), Element);
}

/// @brief  Wrapper function around std::count_if to count the number of times an element
/// satisfying a given predicate occurs in a range
template <typename R, typename UnaryPredicate>
auto count_if(R&& Range, UnaryPredicate P) ->
    typename std::iterator_traits<decltype(std::begin(Range))>::difference_type {
  return std::count_if(std::begin(Range), std::end(Range), P);
}

/// @brief  Wrapper function around std::transform to apply a function to a range and store the
/// result elsewhere
template <typename R, typename OutputIt, typename UnaryPredicate>
OutputIt transform(R&& Range, OutputIt d_first, UnaryPredicate P) {
  return std::transform(std::begin(Range), std::end(Range), d_first, P);
}

//===------------------------------------------------------------------------------------------===//
//     Extra additions to <type_traits>
//===------------------------------------------------------------------------------------------===//

template <typename T, typename = void>
struct is_callable : std::is_function<T> {};

template <typename T>
struct is_callable<
    T, typename std::enable_if<std::is_same<decltype(void(&T::operator())), void>::value>::type>
    : std::true_type {};

/// @}

} // namespace dawn
