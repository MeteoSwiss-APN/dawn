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

using namespace gtclang::dsl;

stencil hori_diff_stencil {
  storage u, out;
  var lap;

  Do {
    vertical_region(k_start, k_end) {
      lap = u(i + 1) + u(i - 1) + u(j + 1) + u(j - 1) - 4.0 * u;
      out = lap(i + 1) + lap(i - 1) + lap(j + 1) + lap(j - 1) - 4.0 * lap;
    }
  }
};
