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

// RUN: %gtclang% %file% -fno-codegen -freport-pass-preprocessor

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;
using namespace gridtools;

// clang-format off
stencil_function TestFun {
  storage foo;

  Do { return foo[i + 1]; }
};

stencil Test01 {
  storage foo, bar;

  void Do() {
    vertical_region(k_start, k_start) {
      foo[i, j + 1, k] = 5;      // EXPECTED: %line%: foo(i, j + 1, k) = 5;
      foo = bar[i];              // EXPECTED: %line%: foo = bar(i);
      foo[i] = bar[i, j];        // EXPECTED: %line%: foo(i) = bar(i, j);
      foo = TestFun(bar[i + 1]); // EXPECTED: %line%: foo = TestFun(bar(i + 1));
    }
  }
};

stencil Test02 {
  gridtools::clang::storage foo, bar;

  void Do() {
    vertical_region(k_start, k_start) 
      foo[i, j + 1, k] = bar; // EXPECTED: %line%: foo(i, j + 1, k) = bar;
  }
};

stencil Test03 {
  clang::storage foo, bar;

  void Do() {
    vertical_region(k_start, k_start) 
      foo[i, j + 1, k] = bar; // EXPECTED: %line%: foo(i, j + 1, k) = bar;
  }
};

// clang-format on
