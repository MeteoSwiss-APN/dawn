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

globals { double global_var = 12; };

stencil_function zero {
  storage a;
  Do { a = 0; }
};

stencil SplitStencil {
  storage intermediate;
  storage out;

  boundary_condition(zero(), intermediate);
  boundary_condition(zero(), out);

  void Do() {
    vertical_region(k_start, k_end) {
      intermediate = out[i + 1];
      out = intermediate[i - 1] + global_var;
    }
  }
};
