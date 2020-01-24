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

#ifndef DAWN_SUPPORT_ITERATORADAPTERS_H
#define DAWN_SUPPORT_ITERATORADAPTERS_H

#include <iterator>
#include <tuple>

namespace dawn {

// enumerate on containers
template <typename T, typename TIter = decltype(std::begin(std::declval<T>())),
          typename = decltype(std::end(std::declval<T>()))>
constexpr auto enumerate(T&& iterable) {
  struct iterator {
    size_t i;
    TIter iter;
    bool operator!=(const iterator& other) const { return iter != other.iter; }
    void operator++() {
      ++i;
      ++iter;
    }
    auto operator*() const { return std::tie(i, *iter); }
  };
  struct iterable_wrapper {
    T iterable;
    auto begin() { return iterator{0, std::begin(iterable)}; }
    auto end() { return iterator{0, std::end(iterable)}; }
  };
  return iterable_wrapper{std::forward<T>(iterable)};
}

template <typename T>
struct reversion_wrapper {
  T& iterable;
};

template <typename T>
auto begin(reversion_wrapper<T> w) {
  return std::rbegin(w.iterable);
}

template <typename T>
auto end(reversion_wrapper<T> w) {
  return std::rend(w.iterable);
}

template <typename T>
reversion_wrapper<T> reverse(T&& iterable) {
  return {iterable};
}

} // namespace dawn

#endif // DAWN_SUPPORT_ITERATORADAPTERS_H
