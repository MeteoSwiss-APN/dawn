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
#include <cstddef>

namespace dawn {

/// @struct AlignedCharArray
/// @brief Helper for building an aligned character array type
///
/// This template is used to explicitly build up a collection of aligned character array types.
template <std::size_t Alignment, std::size_t Size>
struct AlignedCharArray {
  DAWN_ALIGNAS(Alignment) char buffer[Size];
};

namespace internal {

template <typename T1, typename T2 = char, typename T3 = char, typename T4 = char,
          typename T5 = char, typename T6 = char, typename T7 = char, typename T8 = char,
          typename T9 = char, typename T10 = char>
class AlignerImpl {
  T1 t1;
  T2 t2;
  T3 t3;
  T4 t4;
  T5 t5;
  T6 t6;
  T7 t7;
  T8 t8;
  T9 t9;
  T10 t10;

  AlignerImpl() = delete;
};

template <typename T1, typename T2 = char, typename T3 = char, typename T4 = char,
          typename T5 = char, typename T6 = char, typename T7 = char, typename T8 = char,
          typename T9 = char, typename T10 = char>
union SizerImpl {
  char arr1[sizeof(T1)], arr2[sizeof(T2)], arr3[sizeof(T3)], arr4[sizeof(T4)], arr5[sizeof(T5)],
      arr6[sizeof(T6)], arr7[sizeof(T7)], arr8[sizeof(T8)], arr9[sizeof(T9)], arr10[sizeof(T10)];
};

} // namespace internal

/// @brief This union template exposes a suitably aligned and sized character array member which
/// can hold elements of any of up to ten types.
///
/// These types may be arrays, structs, or any other types. The goal is to expose a char array
/// buffer member which can be used as suitable storage for a placement new of any of these types.
/// Support for more than ten types can be added at the cost of more boilerplate.
///
/// @ingroup support
template <typename T1, typename T2 = char, typename T3 = char, typename T4 = char,
          typename T5 = char, typename T6 = char, typename T7 = char, typename T8 = char,
          typename T9 = char, typename T10 = char>
struct AlignedCharArrayUnion
    : AlignedCharArray<alignof(internal::AlignerImpl<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>),
                       sizeof(internal::SizerImpl<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>)> {};

} // namespace dawn
