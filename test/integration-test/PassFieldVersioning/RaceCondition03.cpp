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

// RUN: %gtclang% %file% -fno-codegen -freport-pass-field-versioning -inline=none

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

stencil_function TestFunction {
  storage field_a, field_b;

  Do {
    field_b = field_a;
    field_a = field_b(i + 1);
    return 0.0;
  }
};

stencil Test {
  storage field_a, field_b;

  Do {
    vertical_region(k_start, k_end) {
      TestFunction(field_a, field_b); // EXPECTED_ERROR: unresolvable race-condition in statement
    }
  }
};

int main() {}
