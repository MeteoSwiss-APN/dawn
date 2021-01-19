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
#include "dawn/Support/StringUtil.h"

#include <cstring>

namespace dawn {

/// @brief A switch()-like statement whose cases are string literals.
///
/// The StringSwitch class is a simple form of a switch() statement that determines whether the
/// given string matches one of the given string literals. The template type parameter @p T is the
/// type of the value that will be returned from the string-switch expression. For example,
/// the following code switches on the name of a color in @c argv[i]:
///
/// @code
/// Color color = StringSwitch<Color>(argv[i])
///   .Case("red", Red)
///   .Case("orange", Orange)
///   .Case("yellow", Yellow)
///   .Case("green", Green)
///   .Case("blue", Blue)
///   .Case("indigo", Indigo)
///   .Cases("violet", "purple", Violet)
///   .Default(UnknownColor);
/// @endcode
template <typename T, typename R = T>
class StringSwitch {
  /// The string we are matching.
  const std::string& Str;

  /// The pointer to the result of this switch statement, once known, null before that.
  const T* Result;

public:
  inline explicit StringSwitch(const std::string& S) : Str(S), Result(nullptr) {}

  // StringSwitch is not copyable.
  StringSwitch(const StringSwitch&) = delete;
  void operator=(const StringSwitch&) = delete;

  StringSwitch(StringSwitch&& other) { *this = std::move(other); }
  StringSwitch& operator=(StringSwitch&& other) {
    Str = other.Str;
    Result = other.Result;
    return *this;
  }

  ~StringSwitch() = default;

  // Case-sensitive case matchers
  template <unsigned N>
  inline StringSwitch& Case(const char (&S)[N], const T& Value) {
    DAWN_ASSERT(N);
    if(!Result && N - 1 == Str.size() && (N == 1 || std::memcmp(S, Str.data(), N - 1) == 0)) {
      Result = &Value;
    }
    return *this;
  }

  template <unsigned N>
  inline StringSwitch& EndsWith(const char (&S)[N], const T& Value) {
    DAWN_ASSERT(N);
    if(!Result && Str.size() >= N - 1 &&
       (N == 1 || std::memcmp(S, Str.data() + Str.size() + 1 - N, N - 1) == 0)) {
      Result = &Value;
    }
    return *this;
  }

  template <unsigned N>
  inline StringSwitch& StartsWith(const char (&S)[N], const T& Value) {
    DAWN_ASSERT(N);
    if(!Result && Str.size() >= N - 1 && (N == 1 || std::memcmp(S, Str.data(), N - 1) == 0)) {
      Result = &Value;
    }
    return *this;
  }

  template <unsigned N0, unsigned N1>
  inline StringSwitch& Cases(const char (&S0)[N0], const char (&S1)[N1], const T& Value) {
    return Case(S0, Value).Case(S1, Value);
  }

  template <unsigned N0, unsigned N1, unsigned N2>
  inline StringSwitch& Cases(const char (&S0)[N0], const char (&S1)[N1], const char (&S2)[N2],
                             const T& Value) {
    return Case(S0, Value).Cases(S1, S2, Value);
  }

  template <unsigned N0, unsigned N1, unsigned N2, unsigned N3>
  inline StringSwitch& Cases(const char (&S0)[N0], const char (&S1)[N1], const char (&S2)[N2],
                             const char (&S3)[N3], const T& Value) {
    return Case(S0, Value).Cases(S1, S2, S3, Value);
  }

  template <unsigned N0, unsigned N1, unsigned N2, unsigned N3, unsigned N4>
  inline StringSwitch& Cases(const char (&S0)[N0], const char (&S1)[N1], const char (&S2)[N2],
                             const char (&S3)[N3], const char (&S4)[N4], const T& Value) {
    return Case(S0, Value).Cases(S1, S2, S3, S4, Value);
  }

  template <unsigned N0, unsigned N1, unsigned N2, unsigned N3, unsigned N4, unsigned N5>
  inline StringSwitch& Cases(const char (&S0)[N0], const char (&S1)[N1], const char (&S2)[N2],
                             const char (&S3)[N3], const char (&S4)[N4], const char (&S5)[N5],
                             const T& Value) {
    return Case(S0, Value).Cases(S1, S2, S3, S4, S5, Value);
  }

  template <unsigned N0, unsigned N1, unsigned N2, unsigned N3, unsigned N4, unsigned N5,
            unsigned N6>
  inline StringSwitch& Cases(const char (&S0)[N0], const char (&S1)[N1], const char (&S2)[N2],
                             const char (&S3)[N3], const char (&S4)[N4], const char (&S5)[N5],
                             const char (&S6)[N6], const T& Value) {
    return Case(S0, Value).Cases(S1, S2, S3, S4, S5, S6, Value);
  }

  template <unsigned N0, unsigned N1, unsigned N2, unsigned N3, unsigned N4, unsigned N5,
            unsigned N6, unsigned N7>
  inline StringSwitch& Cases(const char (&S0)[N0], const char (&S1)[N1], const char (&S2)[N2],
                             const char (&S3)[N3], const char (&S4)[N4], const char (&S5)[N5],
                             const char (&S6)[N6], const char (&S7)[N7], const T& Value) {
    return Case(S0, Value).Cases(S1, S2, S3, S4, S5, S6, S7, Value);
  }

  template <unsigned N0, unsigned N1, unsigned N2, unsigned N3, unsigned N4, unsigned N5,
            unsigned N6, unsigned N7, unsigned N8>
  inline StringSwitch& Cases(const char (&S0)[N0], const char (&S1)[N1], const char (&S2)[N2],
                             const char (&S3)[N3], const char (&S4)[N4], const char (&S5)[N5],
                             const char (&S6)[N6], const char (&S7)[N7], const char (&S8)[N8],
                             const T& Value) {
    return Case(S0, Value).Cases(S1, S2, S3, S4, S5, S6, S7, S8, Value);
  }

  template <unsigned N0, unsigned N1, unsigned N2, unsigned N3, unsigned N4, unsigned N5,
            unsigned N6, unsigned N7, unsigned N8, unsigned N9>
  inline StringSwitch& Cases(const char (&S0)[N0], const char (&S1)[N1], const char (&S2)[N2],
                             const char (&S3)[N3], const char (&S4)[N4], const char (&S5)[N5],
                             const char (&S6)[N6], const char (&S7)[N7], const char (&S8)[N8],
                             const char (&S9)[N9], const T& Value) {
    return Case(S0, Value).Cases(S1, S2, S3, S4, S5, S6, S7, S8, S9, Value);
  }

  // Case-insensitive case matchers.
  template <unsigned N>
  inline StringSwitch& CaseLower(const char (&S)[N], const T& Value) {
    if(!Result && equalsLower(Str, S))
      Result = &Value;

    return *this;
  }

  template <unsigned N>
  inline StringSwitch& EndsWithLower(const char (&S)[N], const T& Value) {
    if(!Result && endsWithLower(Str, S))
      Result = &Value;

    return *this;
  }

  template <unsigned N>
  inline StringSwitch& StartsWithLower(const char (&S)[N], const T& Value) {
    if(!Result && startsWithLower(Str, S))
      Result = &Value;

    return *this;
  }

  template <unsigned N0, unsigned N1>
  inline StringSwitch& CasesLower(const char (&S0)[N0], const char (&S1)[N1], const T& Value) {
    return CaseLower(S0, Value).CaseLower(S1, Value);
  }

  template <unsigned N0, unsigned N1, unsigned N2>
  inline StringSwitch& CasesLower(const char (&S0)[N0], const char (&S1)[N1], const char (&S2)[N2],
                                  const T& Value) {
    return CaseLower(S0, Value).CasesLower(S1, S2, Value);
  }

  template <unsigned N0, unsigned N1, unsigned N2, unsigned N3>
  inline StringSwitch& CasesLower(const char (&S0)[N0], const char (&S1)[N1], const char (&S2)[N2],
                                  const char (&S3)[N3], const T& Value) {
    return CaseLower(S0, Value).CasesLower(S1, S2, S3, Value);
  }

  template <unsigned N0, unsigned N1, unsigned N2, unsigned N3, unsigned N4>
  inline StringSwitch& CasesLower(const char (&S0)[N0], const char (&S1)[N1], const char (&S2)[N2],
                                  const char (&S3)[N3], const char (&S4)[N4], const T& Value) {
    return CaseLower(S0, Value).CasesLower(S1, S2, S3, S4, Value);
  }

  inline R Default(const T& Value) const {
    if(Result)
      return *Result;
    return Value;
  }

  inline operator R() const {
    DAWN_ASSERT_MSG(Result, "Fell off the end of a string-switch");
    return *Result;
  }
};

} // namespace dawn
