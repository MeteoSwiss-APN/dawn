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

#include <functional>
#include <iterator>
#include <type_traits>

namespace dawn {

/// @brief class that wraps the value pointed by a IndexRangeIterator,
/// together with the index position within the loop range (IndexRange)
///
template <typename T>
struct IteratorWrap {
private:
  T& t_;
  int idx_;

public:
  IteratorWrap(T& t, int idx) : t_(t), idx_(idx) {}
  T& operator*() { return t_; }
  T* operator->() { return &t_; }

  const T& operator*() const { return t_; }
  const T* operator->() const { return &t_; }
};

/// @brief Iterator class of an IndexRange
/// @tparam cont input STL container
/// @ingroup support
///
template <typename Cont>
struct IndexRangeIterator {
  using T = typename std::conditional<std::is_const<Cont>::value, const typename Cont::value_type,
                                      typename Cont::value_type>::type;
  using iterator =
      typename std::conditional<std::is_const<Cont>::value, typename Cont::const_iterator,
                                typename Cont::iterator>::type;
  using difference_type = typename iterator::difference_type;

  // TODO make this private
private:
  iterator it_;
  iterator begin_;
  iterator end_;
  std::function<bool(T const&)> pred_;
  int idx_ = 0;

public:
  /// @brief constructor
  IndexRangeIterator(Cont& cont, std::function<bool(T const&)> pred)
      : it_(nextValid(cont.begin(), cont.end(), pred)), begin_(cont.begin()), end_(cont.end()),
        pred_(pred) {}

  /// @brief constructor
  IndexRangeIterator(iterator it, iterator begin, iterator end, std::function<bool(T const&)> pred,
                     int index)
      : it_(nextValid(it, end, pred)), begin_(begin), end_(end), pred_(pred), idx_(index) {}

  /// @brief it dereferences the value of an iterator
  /// @returns an IteratorWrap
  T& operator*() { return *it_; }

  /// @returns the next valid iterators, for which the predicate evaluates to true
  static iterator nextValid(iterator it, iterator end, std::function<bool(T const&)> pred) {
    while((it != end) && (!pred(*it)))
      it++;
    return (it);
  }

  void setToEnd() {
    while(it_ != end_)
      this->operator++();
  }

  int idx() const { return idx_; }

  /// @returns the previous valid iterators, for which the predicate evaluates to true
  static iterator prevValid(iterator it, iterator begin, std::function<bool(T const&)> pred) {
    while((it != begin) && (!pred(*it)))
      it--;
    return (it);
  }

  /// @brief increment the iterator
  IndexRangeIterator& operator++() {
    ++it_;
    idx_++;
    it_ = nextValid(it_, end_, pred_);
    return (*this);
  }
  /// @brief increment the iterator
  IndexRangeIterator& operator--() {
    --it_;
    idx_--;
    it_ = prevValid(it_, begin_, pred_);
    return (*this);
  }

  friend typename IndexRangeIterator::difference_type distance(IndexRangeIterator first,
                                                               IndexRangeIterator last) {
    return last.idx() - first.idx();
  }

  /// @brief comparison operators
  friend bool operator!=(IndexRangeIterator l, IndexRangeIterator r) { return l.it_ != r.it_; }
};

/// @brief range class to be used within C++11 for range loops,
/// that accepts predicates to filter elements of the iteration.
///
/// The following example
///
/// @code
///   std::vector<int> v{1,2,3,4,5};
///   IndexRange<std::vector<int>> range(v, [](int const &v){ return v%2;});
///   std::vector<int> res;
///   for(auto it: range) {
///     std::cout << "(" << it.idx() << "," << "it << ")" << std::endl;
///   }
/// @endcode
///
/// generates the output
///
/// @code
/// (0,2)(1,4)
/// @endcode
///
/// @ingroup support
///
template <typename Cont>
struct IndexRange {
  using T = typename Cont::value_type;
  Cont& cont_;
  std::function<bool(T const&)> pred_;
  const size_t size_;

  /// @brief Constructor
  IndexRange(Cont& cont, std::function<bool(T const&)> pred)
      : cont_(cont), pred_(pred), size_(computeSize()) {}

  /// @returns number of elements in the range
  size_t size() const { return size_; }

  /// @returns iterator to the end of the range
  IndexRangeIterator<Cont> end() {
    auto it = IndexRangeIterator<Cont>(cont_, pred_);
    it.setToEnd();
    return it;
  }
  /// @returns iterator to the end of the range
  IndexRangeIterator<const Cont> end() const {
    auto it = IndexRangeIterator<Cont>(cont_, pred_);
    it.setToEnd();
    return it;
  }

  /// @returns iterator to the beginning of the range
  IndexRangeIterator<Cont> begin() { return IndexRangeIterator<Cont>(cont_, pred_); }
  /// @returns iterator to the beginning of the range
  IndexRangeIterator<const Cont> begin() const {
    return IndexRangeIterator<const Cont>(cont_, pred_);
  }

  /// @returns true if range contains no elements (compatible with the predicate)
  bool empty() const { return !(end() != begin()); }

private:
  size_t computeSize() {
    size_t size = 0;
    for(IndexRangeIterator<Cont> it = begin(); it != end(); ++it) {
      size++;
    }
    return size;
  }
};

/// @brief creates an IndexRange from a STL container
/// @param cont input STL container
/// @param pred predicate that filters the elements that will be iterated over by the range
/// @see IndexRange for an example on how to use the index ranges
/// @ingroup support
///
template <typename Cont, typename Fun>
IndexRange<const Cont> makeRange(Cont const& cont, Fun&& pred) {
  return IndexRange<const Cont>{cont, std::forward<Fun>(pred)};
}

template <typename Cont, typename Fun>
IndexRange<Cont> makeRange(Cont& cont, Fun&& pred) {
  return IndexRange<Cont>{cont, std::forward<Fun>(pred)};
}

} // namespace dawn

namespace std {
template <typename Cont>
struct iterator_traits<dawn::IndexRangeIterator<Cont>> {
  using difference_type =
      typename iterator_traits<typename dawn::IndexRangeIterator<Cont>::iterator>::difference_type;
  using value_type =
      typename iterator_traits<typename dawn::IndexRangeIterator<Cont>::iterator>::value_type;
  using pointer =
      typename iterator_traits<typename dawn::IndexRangeIterator<Cont>::iterator>::pointer;
  using iterator_category = typename iterator_traits<
      typename dawn::IndexRangeIterator<Cont>::iterator>::iterator_category;
};

} // namespace std
