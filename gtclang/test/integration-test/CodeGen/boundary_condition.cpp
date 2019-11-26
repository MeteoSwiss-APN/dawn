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
using namespace gtclang::dsl;

stencil_function constant {
  storage a;
  Do { a = 10; }
};

globals { double in_glob = 12; };

stencil split_stencil {
  storage in, out;

  boundary_condition(constant(), in);
  void Do() {
    vertical_region(k_start, k_end) {
      in = out[j + 1];
      out = in[j - 1] + in_glob;
    }
  }
};
