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

#include "dawn/Support/Assert.h"
#include "dawn/Support/Compiler.h"
#include "dawn/Support/STLExtras.h"
#include <array>
#include <vector>

namespace dawn {

/// @brief Represent a constant reference to an array (0 or more elements consecutively in memory)
///
/// It allows various APIs to take consecutive elements easily and conveniently.
///
/// This class does not own the underlying data, it is expected to be used in situations where
/// the data resides in some other buffer, whose lifetime extends past that of the ArrayRef.
/// For this reason, it is not in general safe to store an ArrayRef.
///
/// This is intended to be trivially copyable, so it should be passed by value.
///
/// @ingroup support
template <typename T>
class ArrayRef {
public:
  typedef const T* iterator;
  typedef const T* const_iterator;
  typedef size_t size_type;
  typedef T value_type;

  typedef std::reverse_iterator<iterator> reverse_iterator;

private:
  const T* data_;
  size_type length_;

public:
  /// @name Constructors
  /// @{

  /// @brief Construct an empty ArrayRef
  ArrayRef() : data_(nullptr), length_(0) {}

  /// @brief Construct an ArrayRef from a single element
  ArrayRef(const T& OneElt) : data_(&OneElt), length_(1) {}

  /// @brief Construct an ArrayRef from a pointer and length
  ArrayRef(const T* data, size_t length) : data_(data), length_(length) {}

  /// @brief Construct an ArrayRef from a range
  ArrayRef(const T* begin, const T* end) : data_(begin), length_(end - begin) {}

  /// @brief Construct an ArrayRef from a std::vector
  template <typename A>
  ArrayRef(const std::vector<T, A>& Vec) : data_(Vec.data()), length_(Vec.size()) {}

  /// @brief Construct an ArrayRef from a std::array
  template <size_t N>
  constexpr ArrayRef(const std::array<T, N>& Arr) : data_(Arr.data()), length_(N) {}

  /// @brief Construct an ArrayRef from a C array
  template <size_t N>
  constexpr ArrayRef(const T (&Arr)[N]) : data_(Arr), length_(N) {}

/// @brief Construct an ArrayRef from a std::initializer_list
#if DAWN_GNUC_PREREQ(9, 0, 0)
#pragma GCC diagnostic push
// Probably not an issue, see discussion
// http://lists.llvm.org/pipermail/llvm-dev/2018-September/126078.html
#pragma GCC diagnostic ignored "-Winit-list-lifetime"
#endif
  ArrayRef(const std::initializer_list<T>& Vec)
      : data_(Vec.begin() == Vec.end() ? nullptr : Vec.begin()), length_(Vec.size()) {}
#if DAWN_GNUC_PREREQ(9, 0, 0)
#pragma GCC diagnostic pop
#endif

  /// @brief Construct an ArrayRef<const T*> from ArrayRef<T*>. This uses SFINAE to ensure that
  /// only ArrayRefs of pointers can be converted
  template <typename U>
  ArrayRef(
      const ArrayRef<U*>& A,
      typename std::enable_if<std::is_convertible<U* const*, T const*>::value>::type* = nullptr)
      : data_(A.data()), length_(A.size()) {}

  /// @brief Construct an ArrayRef<const T*> from std::vector<T*>. This uses SFINAE to ensure that
  /// only vectors of pointers can be converted
  template <typename U, typename A>
  ArrayRef(const std::vector<U*, A>& Vec,
           typename std::enable_if<std::is_convertible<U* const*, T const*>::value>::type* = 0)
      : data_(Vec.data()), length_(Vec.size()) {}

  /// @}
  /// @name Simple Operations
  /// @{

  iterator begin() const { return data_; }
  iterator end() const { return data_ + length_; }

  reverse_iterator rbegin() const { return reverse_iterator(end()); }
  reverse_iterator rend() const { return reverse_iterator(begin()); }

  /// @brief Check if the array is empty
  bool empty() const { return length_ == 0; }

  const T* data() const { return data_; }

  /// @brief Get the array size
  size_t size() const { return length_; }

  /// @brief Get the first element
  const T& front() const {
    DAWN_ASSERT(!empty());
    return data_[0];
  }

  /// @brief Get the last element
  const T& back() const {
    DAWN_ASSERT(!empty());
    return data_[length_ - 1];
  }

  /// @brief Allocate copy in Allocator and return ArrayRef<T> to it
  template <typename Allocator>
  ArrayRef<T> copy(Allocator& A) {
    T* Buff = A.template Allocate<T>(length_);
    std::uninitialized_copy(begin(), end(), Buff);
    return ArrayRef<T>(Buff, length_);
  }

  /// @brief Check for element-wise equality
  bool equals(ArrayRef RHS) const {
    if(length_ != RHS.length_)
      return false;
    return std::equal(begin(), end(), RHS.begin());
  }

  /// @brief Chop off the first @p N elements of the array, and keep @p M elements in the array
  ArrayRef<T> slice(size_t N, size_t M) const {
    DAWN_ASSERT_MSG(N + M <= size(), "Invalid specifier");
    return ArrayRef<T>(data() + N, M);
  }

  /// @brief Chop off the first @p N elements of the array.
  ArrayRef<T> slice(size_t N) const { return slice(N, size() - N); }

  /// @brief Drop the first @p N elements of the array.
  ArrayRef<T> drop_front(size_t N = 1) const {
    DAWN_ASSERT_MSG(size() >= N, "Dropping more elements than exist");
    return slice(N, size() - N);
  }

  /// @brief Drop the last \p N elements of the array.
  ArrayRef<T> drop_back(size_t N = 1) const {
    DAWN_ASSERT_MSG(size() >= N, "Dropping more elements than exist");
    return slice(0, size() - N);
  }

  /// @brief Return a copy of *this with the first N elements satisfying the given predicate removed
  template <class PredicateT>
  ArrayRef<T> drop_while(PredicateT Pred) const {
    return ArrayRef<T>(find_if_not(*this, Pred), end());
  }

  /// @brief Return a copy of *this with the first N elements not satisfying the given predicate
  /// removed.
  template <class PredicateT>
  ArrayRef<T> drop_until(PredicateT Pred) const {
    return ArrayRef<T>(find_if(*this, Pred), end());
  }

  /// @brief Return a copy of *this with only the first \p N elements
  ArrayRef<T> take_front(size_t N = 1) const {
    if(N >= size())
      return *this;
    return drop_back(size() - N);
  }

  /// @brief Return a copy of *this with only the last \p N elements
  ArrayRef<T> take_back(size_t N = 1) const {
    if(N >= size())
      return *this;
    return drop_front(size() - N);
  }

  /// @brief Return the first N elements of this Array that satisfy the given predicate
  template <class PredicateT>
  ArrayRef<T> take_while(PredicateT Pred) const {
    return ArrayRef<T>(begin(), find_if_not(*this, Pred));
  }

  /// @brief Return the first N elements of this Array that don't satisfy the given predicate.
  template <class PredicateT>
  ArrayRef<T> take_until(PredicateT Pred) const {
    return ArrayRef<T>(begin(), find_if(*this, Pred));
  }

  /// @}
  /// @name Operator Overloads
  /// @{
  const T& operator[](size_t Index) const {
    DAWN_ASSERT_MSG(Index < length_, "Invalid index!");
    return data_[Index];
  }

  /// @brief Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}" ontinues to select the
  /// move assignment operator.
  template <typename U>
  typename std::enable_if<std::is_same<U, T>::value, ArrayRef<T>>::type&
  operator=(U&& Temporary) = delete;

  /// @brief Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}" continues to select the
  /// move assignment operator.
  template <typename U>
  typename std::enable_if<std::is_same<U, T>::value, ArrayRef<T>>::type&
  operator=(std::initializer_list<U>) = delete;

  /// @}
  /// @name Expensive Operations
  /// @{
  std::vector<T> vec() const { return std::vector<T>(data_, data_ + length_); }

  /// @}
  /// @name Conversion operators
  /// @{
  operator std::vector<T>() const { return std::vector<T>(data_, data_ + length_); }

  /// @}
};

/// @brief Represent a mutable reference to an array (0 or more elements consecutively in memory)
///
/// It allows various APIs to take and modify consecutive elements easily and conveniently.
///
/// This class does not own the underlying data, it is expected to be used in situations where the
/// data resides in some other buffer, whose lifetime extends past that of the MutableArrayRef.
/// For this reason, it is not in general safe to store a MutableArrayRef.
///
/// This is intended to be trivially copyable, so it should be passed by value.
///
/// @ingroup support
template <typename T>
class MutableArrayRef : public ArrayRef<T> {
public:
  typedef T* iterator;

  typedef std::reverse_iterator<iterator> reverse_iterator;

  /// @brief Construct an empty MutableArrayRef
  MutableArrayRef() : ArrayRef<T>() {}

  /// @brief Construct an MutableArrayRef from a single element
  MutableArrayRef(T& OneElt) : ArrayRef<T>(OneElt) {}

  /// @brief Construct an MutableArrayRef from a pointer and length
  MutableArrayRef(T* data, size_t length) : ArrayRef<T>(data, length) {}

  /// @brief Construct an MutableArrayRef from a range
  MutableArrayRef(T* begin, T* end) : ArrayRef<T>(begin, end) {}

  /// @brief Construct a MutableArrayRef from a std::vector
  MutableArrayRef(std::vector<T>& Vec) : ArrayRef<T>(Vec) {}

  /// @brief Construct an ArrayRef from a std::array
  template <size_t N>
  constexpr MutableArrayRef(std::array<T, N>& Arr) : ArrayRef<T>(Arr) {}

  /// @brief Construct an MutableArrayRef from a C array
  template <size_t N>
  constexpr MutableArrayRef(T (&Arr)[N]) : ArrayRef<T>(Arr) {}

  T* data() const { return const_cast<T*>(ArrayRef<T>::data()); }

  iterator begin() const { return data(); }
  iterator end() const { return data() + this->size(); }

  reverse_iterator rbegin() const { return reverse_iterator(end()); }
  reverse_iterator rend() const { return reverse_iterator(begin()); }

  /// @brief Get the first element
  T& front() const {
    DAWN_ASSERT(!this->empty());
    return data()[0];
  }

  /// @brief Get the last element
  T& back() const {
    DAWN_ASSERT(!this->empty());
    return data()[this->size() - 1];
  }

  /// @brief Chop off the first N elements of the array, and keep M elements in the array
  MutableArrayRef<T> slice(size_t N, size_t M) const {
    DAWN_ASSERT_MSG(N + M <= this->size(), "Invalid specifier");
    return MutableArrayRef<T>(this->data() + N, M);
  }

  /// @brief Chop off the first N elements of the array
  MutableArrayRef<T> slice(size_t N) const { return slice(N, this->size() - N); }

  /// @brief Drop the first \p N elements of the array
  MutableArrayRef<T> drop_front(size_t N = 1) const {
    DAWN_ASSERT_MSG(this->size() >= N, "Dropping more elements than exist");
    return slice(N, this->size() - N);
  }

  MutableArrayRef<T> drop_back(size_t N = 1) const {
    DAWN_ASSERT_MSG(this->size() >= N, "Dropping more elements than exist");
    return slice(0, this->size() - N);
  }

  /// @brief Return a copy of *this with the first N elements satisfying the given predicate removed
  template <class PredicateT>
  MutableArrayRef<T> drop_while(PredicateT Pred) const {
    return MutableArrayRef<T>(find_if_not(*this, Pred), end());
  }

  /// @brief Return a copy of *this with the first N elements not satisfying the given predicate
  /// removed
  template <class PredicateT>
  MutableArrayRef<T> drop_until(PredicateT Pred) const {
    return MutableArrayRef<T>(find_if(*this, Pred), end());
  }

  /// @brief Return a copy of *this with only the first \p N elements
  MutableArrayRef<T> take_front(size_t N = 1) const {
    if(N >= this->size())
      return *this;
    return drop_back(this->size() - N);
  }

  /// @brief Return a copy of *this with only the last \p N elements
  MutableArrayRef<T> take_back(size_t N = 1) const {
    if(N >= this->size())
      return *this;
    return drop_front(this->size() - N);
  }

  /// @brief Return the first N elements of this Array that satisfy the given predicate
  template <class PredicateT>
  MutableArrayRef<T> take_while(PredicateT Pred) const {
    return MutableArrayRef<T>(begin(), find_if_not(*this, Pred));
  }

  /// @brief Return the first N elements of this Array that don't satisfy the given predicate
  template <class PredicateT>
  MutableArrayRef<T> take_until(PredicateT Pred) const {
    return MutableArrayRef<T>(begin(), find_if(*this, Pred));
  }

  /// @}
  /// @name Operator Overloads
  /// @{
  T& operator[](size_t Index) const {
    DAWN_ASSERT_MSG(Index < this->size(), "Invalid index!");
    return data()[Index];
  }
};

/// @brief This is a MutableArrayRef that owns its array.
///
/// @ingroup support
template <typename T>
class OwningArrayRef : public MutableArrayRef<T> {
public:
  OwningArrayRef() {}
  OwningArrayRef(size_t Size) : MutableArrayRef<T>(new T[Size], Size) {}
  OwningArrayRef(ArrayRef<T> Data) : MutableArrayRef<T>(new T[Data.size()], Data.size()) {
    std::copy(Data.begin(), Data.end(), this->begin());
  }
  OwningArrayRef(OwningArrayRef&& Other) { *this = Other; }
  OwningArrayRef& operator=(OwningArrayRef&& Other) {
    delete[] this->data();
    this->MutableArrayRef<T>::operator=(Other);
    Other.MutableArrayRef<T>::operator=(MutableArrayRef<T>());
    return *this;
  }
  ~OwningArrayRef() { delete[] this->data(); }
};

/// @addtogroup support
/// @{

/// @brief Construct an ArrayRef from a single element
template <typename T>
ArrayRef<T> makeArrayRef(const T& OneElt) {
  return OneElt;
}

/// @brief Construct an ArrayRef from a pointer and length
template <typename T>
ArrayRef<T> makeArrayRef(const T* data, size_t length) {
  return ArrayRef<T>(data, length);
}

/// @brief Construct an ArrayRef from a range
template <typename T>
ArrayRef<T> makeArrayRef(const T* begin, const T* end) {
  return ArrayRef<T>(begin, end);
}

/// @brief Construct an ArrayRef from a std::vector.
template <typename T>
ArrayRef<T> makeArrayRef(const std::vector<T>& Vec) {
  return Vec;
}

/// @brief Construct an ArrayRef from an std::initializer_list.
template <typename T>
ArrayRef<T> makeArrayRef(const std::initializer_list<T>& Vec) {
  return Vec;
}

/// @brief Construct an ArrayRef from a std::basic_string<T> .
template <typename T>
ArrayRef<T> makeArrayRef(const std::basic_string<T>& Str) {
  return ArrayRef<T>(Str.data(), Str.size());
}

/// @brief Construct an ArrayRef from an ArrayRef (no-op) (const)
template <typename T>
ArrayRef<T> makeArrayRef(const ArrayRef<T>& Vec) {
  return Vec;
}

/// @brief Construct an ArrayRef from an ArrayRef (no-op)
template <typename T>
ArrayRef<T>& makeArrayRef(ArrayRef<T>& Vec) {
  return Vec;
}

/// @brief Construct an ArrayRef from a C array.
template <typename T, size_t N>
ArrayRef<T> makeArrayRef(const T (&Arr)[N]) {
  return ArrayRef<T>(Arr);
}

/// @brief ArrayRef Comparison Operators
template <typename T>
inline bool operator==(ArrayRef<T> LHS, ArrayRef<T> RHS) {
  return LHS.equals(RHS);
}

/// @brief ArrayRef Comparison Operators
template <typename T>
inline bool operator!=(ArrayRef<T> LHS, ArrayRef<T> RHS) {
  return !(LHS == RHS);
}

/// @}

} // namespace dawn

/// @}
