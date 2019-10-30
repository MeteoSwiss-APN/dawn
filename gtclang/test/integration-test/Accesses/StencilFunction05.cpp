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

// RUN: %gtclang% %file% -fno-codegen -freport-accesses

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

stencil_function foo {
  direction dir;
  storage in;

  Do {
    double val = 0.0;
    if(in > 0.0)
      val = in(dir + 1);
    else
      val = in(dir - 1);
    return val;
  }
};

stencil Test {
  storage field_a, field_b, field_c;

  Do {
    vertical_region(k_start, k_end) {
      field_a = foo(i, field_b); // EXPECTED_ACCESSES: R:field_b:<[(-1,1),(0,0),(0,0)]>
    }
  }
};

int main() {}
