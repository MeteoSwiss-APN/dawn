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

globals {
  int var_runtime;        // == 1
  double var_default = 2; // == 2
  bool var_compiletime;   // == true
};

stencil globals_stencil {
  storage in, out;

  Do {
    vertical_region(k_start, k_end) {
      if(var_compiletime)                     // true
        out = in + var_runtime + var_default; // 1 + 2
    }
  }
};
