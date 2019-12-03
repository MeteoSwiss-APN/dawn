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

stencil lap {
  storage in, out;
  var tmp;

  Do {
    vertical_region(k_start, k_end) {
      tmp = in[j - 2] + in[j + 2] + in[i - 2] + in[i + 2];
      out = tmp[j - 1] + tmp[j + 1] + tmp[i - 1] + tmp[i + 1];
    }
  }
};
