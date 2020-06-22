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

    using namespace std;
    ss << start_;
    auto it = begin(range);
    const std::size_t size = distance(begin(range), end(range));
    std::size_t i = 0;

    bool initialized = false;
    for(; it != end(range); ++it, ++i) {
      if(!stringify(*it).empty() || !ignoreIfEmpty_) {
        ss << stringify(*it);
        initialized = true;
      }
      if(i != size - 1 && ((!stringify(*next(it)).empty() && initialized) || !ignoreIfEmpty_)) {
        ss << delim_;
      }
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

/// @brief Checks if two strings are equal (case insensitive)
/// @ingroup support
bool equalsLower(const std::string& a, const std::string& b);

/// @brief Checks if str begins with match (case insensitive)
/// @ingroup support
bool startWithLower(std::string str, std::string match);

/// @brief Checks if str ends with match (case insensitive)
/// @ingroup support
bool endsWithLower(std::string str, std::string match);

} // namespace dawn
