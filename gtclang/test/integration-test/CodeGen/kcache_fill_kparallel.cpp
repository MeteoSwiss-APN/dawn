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

stencil kcache_fill_kparallel {
  storage in, out;

  Do {
    vertical_region(k_start, k_start + 1) {
      out = in + in[k + 1];
    }
    vertical_region(k_start + 1, k_end - 1) {
      out = in + in[k - 1] + in[k + 1];
    }
    vertical_region(k_end - 1, k_end) {
      out = in + in[k - 1];
    }
  }
};
