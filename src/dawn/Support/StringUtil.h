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

#ifndef DAWN_SUPPORT_STRINGUTIL_H
#define DAWN_SUPPORT_STRINGUTIL_H

#include "dawn/Support/Printing.h"
#include "dawn/Support/STLExtras.h"
#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>

namespace dawn {

namespace internal {

template <class T>
struct StdToString {
  std::string operator()(const T& t) { return std::to_string(t); }
};

template <>
struct StdToString<std::string> {
  std::string operator()(const std::string& t) { return t; }
};

template <>
struct StdToString<const char*> {
  std::string operator()(const char* t) { return t; }
};

} // namespace internal

/// @brief Converts range based containers to string
///
/// @b Example:
/// @code
///   std::vector<int> v = {1,2,3,4,5};
///   std::cout << RangeToString()(v); // [1, 2, 3, 4, 5]
/// @endcode
///
/// @ingroup support
class RangeToString {
  const char* delim_;
  const char* start_;
  const char* end_;
  const bool ignoreIfEmpty_;

public:
  /// @brief Convert each element to string using `std::to_string`
  struct DefaultStringifyFunctor {
    template <class T>
    std::string operator()(const T& t) {
      return internal::StdToString<T>()(t);
    }
  };

  RangeToString(const char* delim = ", ", const char* start = "[", const char* end = "]",
                const bool ignoreIfEmpty = false)
      : delim_(delim), start_(start), end_(end), ignoreIfEmpty_(ignoreIfEmpty) {}

  /// @brief Convert a `Range` to string (elements of the Range are converted to a string via
  /// `std::to_string`)
  ///
  /// Optionally, an already existing `stringstream` can be provided in which case the `range` is
  /// directly streamed into the stream and an empty string is returned.
  template <class Range>
  inline std::string operator()(Range&& range) {
    return this->operator()(std::forward<Range>(range), DefaultStringifyFunctor());
  }

  /// @brief Convert a `Range` to string where each element will be converted to string using the
  /// `stringify` functor
  ///
  /// Optionally, an already existing `stringstream` can be provided in which case the `range` is
  /// directly streamed into the stream and an empty string is returned.
  template <class Range, class StrinfigyFunctor>
  inline std::string operator()(Range&& range, StrinfigyFunctor&& stringify) {
    std::stringstream ss;

    ss << start_;
    auto it = std::begin(range), end = std::end(range);
    const std::size_t size = std::distance(it, end);
    std::size_t i = 0;

    for(; it != end; ++it, ++i) {
      if(!stringify(*it).empty() || !ignoreIfEmpty_) {
        ss << stringify(*it);
      }
      if(i != size - 1 && (!stringify(*std::next(it)).empty() || !ignoreIfEmpty_))
        ss << delim_;
    }
    ss << end_;

    return ss.str();
  }
};

/// @brief Indent a string by the specified number of spaces
/// @ingroup core
///
/// @ingroup support
extern std::string indent(const std::string& string, int amount = 2);

/// @brief Convert a decimal integer to an ordinal string
///
/// @b Example
/// @code
///   std::string one = intToOrdinal(1); // == "1st"
/// @endcode
///
/// @ingroup support
extern std::string decimalToOrdinal(int dec);

/// @}

} // namespace dawn

#endif
