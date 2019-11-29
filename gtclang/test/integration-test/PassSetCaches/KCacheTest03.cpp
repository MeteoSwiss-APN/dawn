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

// RUN: %gtclang% %file% -fno-codegen -freport-pass-set-caches
// EXPECTED: PASS: PassSetCaches: Test: MS0: tmp:cache_type::k:flush:\[0,0\]
// EXPECTED: PASS: PassSetCaches: Test: MS1: tmp:cache_type::k:fill

#include "gtclang_dsl_defs/gtclang_dsl.hpp"

using namespace gridtools::clang;

stencil Test {
  storage in, out;
  storage a, b, c;
  var tmp;

  Do {
    vertical_region(k_start, k_end) {
      // --- MS0 ---
      tmp = in;

      b = a;

      // --- MS1 ---
      c = b(k + 1);
      c = b(k - 1);

      out = tmp;
    }
  }
};

int main() {}
