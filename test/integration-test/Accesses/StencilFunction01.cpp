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

// RUN: %gtclang% %file% -fno-codegen -freport-accesses -inline=none

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

stencil_function TestFunctionReturn {
  storage in;

  Do {
    return in(i + 1, j + 1, k + 1); // EXPECTED_ACCESSES: R:in:<0,1,0,1,0,1>
  }
};

stencil_function TestFunctionByRef {
  storage out, in;

  Do {
    out = in(i + 1, j + 1, k + 1); // EXPECTED_ACCESSES:W:out:<0,0,0,0,0,0> %and% R:in:<0,1,0,1,0,1>
  }
};

struct Test : public stencil {
  using stencil::stencil;

  storage field_a, field_b;

  Do {
    vertical_region(k_start, k_end) {
      field_a = TestFunctionReturn(field_b); // EXPECTED_ACCESSES: W:field_a:<0,0,0,0,0,0> %and% R:field_b:<0,1,0,1,0,1>
      TestFunctionByRef(field_a, field_b);   // EXPECTED_ACCESSES: W:field_a:<0,0,0,0,0,0> %and% R:field_b:<0,1,0,1,0,1>
    }
  }
};

int main() {}
