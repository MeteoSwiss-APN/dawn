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

stencil local_kcache {
  storage out_a, out_b, out_c, out_d;
  var a, b, c, d;

  Do {
    vertical_region(k_start, k_start) {
      c = 2.1;
      d = 3.1;
      out_c = c;
      out_d = d;
    }
    vertical_region(k_start + 1, k_start + 1) {
      d = 4.1;
      out_d = d;
      out_c = c;
    }
    vertical_region(k_start + 2, k_end) {
      c = c[k - 1] * 1.1;
      d = d[k - 1] * 1.1 + d[k - 2] * 1.2;
      out_c = c;
      out_d = d;
    }

    vertical_region(k_end, k_end) {
      a = 2.1;
      b = 3.1;
      out_a = a;
      out_b = b;
    }
    vertical_region(k_end - 1, k_end - 1) {
      b = 4.1;
      out_b = b;
      out_a = a;
    }
    vertical_region(k_end - 2, k_start) {
      a = a[k + 1] * 1.1;
      b = b[k + 1] * 1.1 + b[k + 2] * 1.2;
      out_a = a;
      out_b = b;
    }
  }
};
