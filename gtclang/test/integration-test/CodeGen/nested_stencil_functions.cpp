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

stencil_function delta {
  offset off;
  storage in;

  Do { return in(off)-in; }
};

////
//// Test 4
////

stencil_function sum {
  storage s1, s2;

  Do { return s1 + s2; }
};

stencil_function delta_sum {
  offset off1;
  offset off2;
  storage in0;

  Do { return sum(delta(off1, in0), delta(off2, in0)); }
};

stencil test_04_stencil {
  storage in, out;

  Do {
    vertical_region(k_start, k_end)
        out = delta_sum(i + 1, j + 1, in);
  }
};

////
//// Test 5
////

stencil test_05_stencil {
  storage in, out;

  Do {
    vertical_region(k_start, k_end)
        out = delta(i + 1, delta(j + 1, delta(i + 1, in)));
  }
};
