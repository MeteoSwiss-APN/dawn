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
  int var_runtime = 1; // == 1
  int var_compiletime; // == 2
};

//
// Test 1
//
stencil test_01_stencil {
  storage in, out;

  Do {
    if(var_runtime == 1)
      vertical_region(k_start, k_end) {
        out = in + var_runtime;
      }
  }
};

//
// Test 2
//
stencil test_02_stencil {
  storage in, out;

  Do {
    if(var_compiletime == 2)
      vertical_region(k_start, k_end)
          out = in + var_compiletime;
  }
};

//
// Test 3
//
stencil test_03_stencil {
  storage in, out;

  Do {
    if(var_runtime == 1)
      if(var_compiletime == 2)
        vertical_region(k_start, k_end)
            out = in + var_runtime + var_compiletime;
  }
};

//
// Test 4
//
stencil test_04_stencil {
  storage in, out;

  Do {
    if(var_compiletime == 2)
      if(var_compiletime != 1) {
        vertical_region(k_start, k_end)
            out = 0.0;
        if(var_compiletime != 1) {
          vertical_region(k_start, k_end)
              out += 2 + in;
        }
      }
  }
};

//
// Test 5
//
stencil test_05_stencil {
  storage in, out;

  Do {
    if(var_compiletime == 2) {
      double some_var = 5.0;
      if(var_runtime < some_var)
        vertical_region(k_start, k_end)
            out = 2 * in;
    }
  }
};

//
// Test 6
//
stencil test_06_stencil {
  storage in, out;

  Do {
    if(var_compiletime == 2) {
      double some_var = 5.0;
      if(var_compiletime < some_var)
        vertical_region(k_start, k_end)
            out = 2 * in;
    }
  }
};

//
// Test 7
//
stencil test_07_stencil {
  storage in, out;

  Do {
    if(var_compiletime == 2) {
      double some_var = 5.0;
      double some_other_var = var_compiletime;

      some_var += 1.0;

      if((var_compiletime + some_var + some_other_var) == 10)
        vertical_region(k_start, k_end)
            out = 2 * in;
    }
  }
};

//
// Test 8
//
stencil test_08_stencil {
  storage in, out;
  var tmp;

  Do {
    if(var_compiletime == 2) {
      vertical_region(k_start, k_end)
          tmp = 2 * in;
    }
    if(var_compiletime == 2) {
      vertical_region(k_start, k_end)
          out = 2 * tmp;
    }
  }
};

//
// Test 9
//
stencil test_09_stencil_call {
  storage in, out;

  Do {
    if(var_compiletime == 2) {
      vertical_region(k_start, k_end)
          out = 2 * in;
    }
  }
};

stencil test_09_stencil {
  storage in, out;

  Do {
    if(var_compiletime == 2) {
      test_09_stencil_call(in, out);
    }
  }
};
