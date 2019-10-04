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
// EXPECTED: PASS: PassSetCaches: Test: MS0: tmp:cache_type::k:epflush:[-3,0]
// EXPECTED: PASS: PassSetCaches: Test: MS1: tmp:cache_type::k:bpfill:[0,3]

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

stencil Test {
  storage a, b, c;
  var tmp;

  Do {
    // MS0
    vertical_region(k_start, k_start)
        tmp = a;

    vertical_region(k_start + 1, k_end) {
      tmp = a * 2;
      b = tmp(k - 1);
    }

    // MS1
    vertical_region(k_end - 3, k_end - 3) {
      c = tmp[k + 3] + tmp[k + 2] + tmp[k + 1];
    }
    vertical_region(k_end - 4, k_start) {
      tmp = b;
      c = tmp[k + 1];
    }
  }
};

int main() {}
