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
// EXPECTED: PASS: PassSetCaches: Test1: MS0: in:cache_type::k:fill
// EXPECTED: PASS: PassSetCaches: Test2: MS0: out:cache_type::k:fill_and_flush
// EXPECTED: PASS: PassSetCaches: Test2: MS0: in:cache_type::k:fill

#include "gtclang_dsl_defs/gtclang_dsl.hpp"

using namespace gridtools::clang;

stencil Test1 {
  storage in, out;

  Do {
    vertical_region(k_start, k_end) {
      out = in + in(k + 1);
    }
  }
};

stencil Test2 {
  storage in, out;

  Do {
    vertical_region(k_start, k_end) {
      out += in + in(k + 1);
    }
  }
};

int main() {}
