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

stencil compute_extent_test_stencil {
  storage u, out;

  var tmp0, tmp1, tmp2, tmp3;
  Do {
    vertical_region(k_start, k_end) {
      tmp0 = u[i + 1] + u[i - 1];
      tmp1 = tmp0[j + 1] + tmp0[i - 1];
      tmp2 = tmp1[i - 1] + tmp0[j - 1];
      tmp3 = tmp0[i + 2] + tmp0[j - 1];
      out = tmp3[i + 1] + tmp2[j - 1];
    }
  }
};
