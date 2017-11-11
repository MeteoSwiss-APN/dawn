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

// RUN: %gtclang% %file% -fno-codegen -fuse-kcaches -freport-pass-set-caches
// EXPECTED: PASS: PassSetCaches: Test1: MS0: in:K:fill
// EXPECTED: PASS: PassSetCaches: Test2: MS0: out:K:fill_and_flush
// EXPECTED: PASS: PassSetCaches: Test2: MS0: in:K:fill

#include "gridtools/clang_dsl.hpp"

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
