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
#include <list>
#include <numeric>
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

// reverse
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

// zip
template <typename T1, typename T2, typename T1Iter = decltype(std::begin(std::declval<T1>())),
          typename T2Iter = decltype(std::begin(std::declval<T2>())),
          typename = decltype(std::end(std::declval<T1>())),
          typename = decltype(std::end(std::declval<T2>()))>
constexpr auto zip(T1&& iterable1, T2&& iterable2) {
  struct iterator {
    T1Iter iter1;
    T2Iter iter2;
    bool operator!=(const iterator& other) const {
      return (iter1 != other.iter1) && (iter2 != other.iter2);
    }
    void operator++() {
      ++iter1;
      ++iter2;
    }
    auto operator*() const { return std::tie(*iter1, *iter2); }
  };
  struct iterable_wrapper {
    T1 iterable1;
    T2 iterable2;
    auto begin() { return iterator{std::begin(iterable1), std::begin(iterable2)}; }
    auto end() { return iterator{std::begin(iterable1), std::end(iterable2)}; }
  };
  return iterable_wrapper{std::forward<T1>(iterable1), std::forward<T2>(iterable2)};
}

template <typename Map>
bool compareMapValues(const Map& map1, const Map& map2) {
  // TODO This only works when Map has a value_type. Could use enable_if and overload later.
  if(map1.size() != map2.size())
    return false;

  for(auto& iter1 : map1) {
    auto iter2 = map2.find(iter1.first);
    if(iter2 == map2.end() || !(iter1.second == iter2->second)) {
      return false;
    }
  }

  return true;
}

} // namespace dawn

#endif // DAWN_SUPPORT_ITERATORADAPTERS_H
