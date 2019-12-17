//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _       _
//                        | |     | |
//                    __ _| |_ ___| | __ _ _ __   __ _
//                   / _` | __/ __| |/ _` | '_ \ / _` |
//                  | (_| | || (__| | (_| | | | | (_| |
//                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
//                    __/ |                       __/ |
//                   |___/                       |___/
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#pragma once

#include <initializer_list>

namespace gtclang {
namespace dsl {

/**
 * @brief Defintion of a vertical interval
 * @ingroup gtclang_dsl
 */
struct interval {
  interval operator+(int) { return *this; }
  interval operator-(int) { return *this; }
};

/**
 * Lowest k-level
 * @ingroup gtclang_dsl
 */
static interval k_start;

/**
 * Highest k-level
 * @ingroup gtclang_dsl
 */
static interval k_end;

struct interval0 {
  template <typename T>
  interval0(T...);
};

struct interval1 {
  template <typename T>
  interval1(T...);
};

struct interval2 {
  template <typename T>
  interval2(T...);
};

struct interval3 {
  template <typename T>
  interval3(T...);
};

struct interval4 {
  template <typename T>
  interval4(T...);
};

struct interval5 {
  template <typename T>
  interval5(T...);
};

struct interval6 {
  template <typename T>
  interval6(T...);
};

struct interval7 {
  template <typename T>
  interval7(T...);
};

struct interval8 {
  template <typename T>
  interval8(T...);
};

struct interval9 {
  template <typename T>
  interval9(T...);
};

struct interval10 {
  template <typename T>
  interval10(T...);
};

struct interval11 {
  template <typename T>
  interval11(T...);
};

struct interval12 {
  template <typename T>
  interval12(T...);
};

struct interval13 {
  template <typename T>
  interval13(T...);
};

struct interval14 {
  template <typename T>
  interval14(T...);
};

struct interval15 {
  template <typename T>
  interval15(T...);
};

struct interval16 {
  template <typename T>
  interval16(T...);
};

struct interval17 {
  template <typename T>
  interval17(T...);
};

struct interval18 {
  template <typename T>
  interval18(T...);
};

struct interval19 {
  template <typename T>
  interval19(T...);
};
} // namespace dsl
} // namespace gtclang
