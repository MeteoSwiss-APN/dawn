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

#include "gtclang_dsl_defs/gtclang_dsl.hpp"

using namespace gtclang::dsl;

stencil_function TestFunctionReturn {
  storage in;

  Do {
    return in(i + 1, j + 1, k + 1); // EXPECTED_ACCESSES: R:in:<[(1,1),(1,1),(1,1)]>
  }
};

stencil_function TestFunctionByRef {
  storage out, in;

  Do {
    out = in(i + 1, j + 1, k + 1); // EXPECTED_ACCESSES:W:out:<[<no_horizontal_extent>,(0,0)]> %and% R:in:<[(1,1),(1,1),(1,1)]>
  }
};

//==================================================================================================
// costodo: fix extents again
//==================================================================================================
struct Test : public stencil {
  using stencil::stencil;

  storage field_a, field_b;

  Do {
    vertical_region(k_start, k_end) {
      field_a = TestFunctionReturn(field_b); // EXPECTED_ACCESSES: W:field_a:<[<no_horizontal_extent>,(0,0)]> %and% R:field_b:<[(1,1),(1,1),(1,1)]>
      TestFunctionByRef(field_a, field_b);   // EXPECTED_ACCESSES: W:field_a:<[<no_horizontal_extent>,(0,0)]> %and% R:field_b:<[(1,1),(1,1),(1,1)]>
    }
  }
};

int main() {}
