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
  storage in, out1, out2;

  var u;
  Do {
    vertical_region(k_start + 2, k_start + 3) u = in;
    vertical_region(k_start + 2, k_start + 10) out1 = u[k + 2] + u[k + 1];
    vertical_region(k_start, k_start + 8) out2 = u[k + 6];
  }
};
