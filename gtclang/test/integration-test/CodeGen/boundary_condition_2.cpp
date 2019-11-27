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
//===-------------------------------------------------------------------------------------------===//
#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gridtools::clang;

stencil_function to_field {
  storage a, b;
  Do { a = b; }
};

globals { double in_glob = 12; };

stencil split_stencil {
  storage in, out, bcfield;

  boundary_condition(to_field(), in, bcfield);
  void Do() {
    vertical_region(k_start, k_end) {
      in = out[j + 1];
      out = in[j - 1] + in_glob;
    }
  }
};
