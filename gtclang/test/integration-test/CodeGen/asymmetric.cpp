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

stencil asymmetric_stencil {
  storage in, out;

  var tmp;

  Do {
    vertical_region(k_start, k_start) {
      tmp = in[i + 1] + in[j - 2];
      out = 1;
    }
    vertical_region(k_start + 1, k_end) {
      tmp = in[i - 3] + in[j + 1];
      out = tmp[k - 1];
    }
  }
};
