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

#include "gridtools/clang_dsl.hpp"
using namespace gridtools::clang;

globals {
  int var1 = 1;
  bool var2;
};

stencil_function fn {
  offset dim;
  storage in, out;
  Do {
    if(var1 == 1) {
      out = in[dim + 1];
    } else {
      out = in[dim - 1];
    }
  }
};

stencil conditional_stencil {
  storage in, out;

  Do {
    vertical_region(k_start, k_end) {
      if(var2) {
        fn(i, in, out);
      } else {
        fn(j, in, out);
      }
    }
  }
};
