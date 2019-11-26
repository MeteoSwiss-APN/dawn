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

#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gridtools::clang;

stencil kcache_epflush {
  storage in, out;
  var b, tmp;

  Do {
    // MS0
    vertical_region(k_start, k_start)
        tmp = in;

    vertical_region(k_start + 1, k_end) {
      tmp = in * 2;
      b = tmp[k - 1];
    }

    // MS1
    vertical_region(k_end, k_end - 2) {
      out = 2.2;
    }
    vertical_region(k_end - 3, k_end - 3) {
      out = tmp[k + 3] + tmp[k + 2] + tmp[k + 1];
    }
    vertical_region(k_end - 4, k_start) {
      tmp = b;
      out = tmp[k + 1];
    }
  }
};
