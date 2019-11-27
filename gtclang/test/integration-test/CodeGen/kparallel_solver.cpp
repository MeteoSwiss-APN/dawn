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

stencil kparallel_solver {
  storage d, a, b, c;

  Do {
    vertical_region(k_start, k_start) {
      c = a[k + 1];
      d = b * 2.1;
    }
    vertical_region(k_start + 1, k_end) {
      c = a[k - 1];
      d = b * 2.0 - 1;
    }
    vertical_region(k_end - 1, k_start) {
      d -= d + a;
    }
  }
};
