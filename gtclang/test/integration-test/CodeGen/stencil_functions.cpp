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

//
// Test 1
//

stencil test_01_stencil {
  storage in, out;

  Do {
    vertical_region(k_start, k_end)
        out = delta(i + 1, in);
  }
};

//
// Test 2
//

stencil test_02_stencil {
  storage in, out;

  Do {
    vertical_region(k_start, k_end)
        out = delta(i + 1, in) + delta(j + 1, in);
  }
};

////
//// Test 3
////

stencil_function delta_nested {
  offset off;
  storage in;

  Do { return delta(off, in); }
};

stencil test_03_stencil {
  storage in, out;

  Do {
    vertical_region(k_start, k_end)
        out = delta_nested(i + 1, in);
  }
};

////
//// Test 6
////

stencil_function layer_1_ret {
  storage in;

  Do { return in; };
};

stencil_function layer_2_ret {
  storage in;

  Do {
    return layer_1_ret(in);
  }
};

stencil_function layer_3_ret {
  storage in;

  Do { return layer_2_ret(in); }
};

stencil test_06_stencil {
  storage in, out;

  Do {
    vertical_region(k_start, k_end)
        out = layer_3_ret(in);
  }
};

////
//// Test 7
////

stencil_function layer_1_no_ret {
  storage out, in;

  Do { out = in; };
};

stencil_function layer_2_no_ret {
  storage out, in;

  Do { layer_1_no_ret(out, in); }
};

stencil_function layer_3_no_ret {
  storage out, in;

  Do { layer_2_no_ret(out, in); }
};

stencil test_07_stencil {
  storage in, out;

  Do {
    vertical_region(k_start, k_end)
        layer_3_no_ret(out, in);
  }
};
