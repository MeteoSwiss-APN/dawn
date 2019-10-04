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

stencil_function TestFunctionDirection {
  direction dir;
  storage in;

  Do { return in(dir + 1); }
};

stencil_function TestFunctionOffset {
  offset off;
  storage in;

  Do { return in(off + 1); }
};

//==================================================================================================
// costodo: fix extents again
//==================================================================================================
stencil Test {
  storage field_a, field_b;

  Do {
    vertical_region(k_start, k_end) {
      field_a = TestFunctionDirection(i, field_b);                      // EXPECTED_ACCESSES: W:field_a:<0,0,0,0,0,0> %and% R:field_b:<1,1,0,0,0,0>
      field_a = TestFunctionDirection(i, field_b(i + 1, j + 1, k + 1)); // EXPECTED_ACCESSES: W:field_a:<0,0,0,0,0,0> %and% R:field_b:<2,2,1,1,1,1>
      field_a = TestFunctionDirection(i, field_b(i - 1));               // EXPECTED_ACCESSES: W:field_a:<0,0,0,0,0,0> %and% R:field_b:<0,0,0,0,0,0>
      field_a = TestFunctionDirection(j, field_b(i - 1));               // EXPECTED_ACCESSES: W:field_a:<0,0,0,0,0,0> %and% R:field_b:<-1,-1,1,1,0,0>
      field_a = TestFunctionDirection(k, field_b(i - 1, k - 1));        // EXPECTED_ACCESSES: W:field_a:<0,0,0,0,0,0> %and% R:field_b:<-1,-1,0,0,0,0>

      field_a = TestFunctionOffset(i, field_b);                      // EXPECTED_ACCESSES: W:field_a:<0,0,0,0,0,0> %and% R:field_b:<1,1,0,0,0,0>
      field_a = TestFunctionOffset(i, field_b(i + 1, j + 1, k + 1)); // EXPECTED_ACCESSES: W:field_a:<0,0,0,0,0,0> %and% R:field_b:<2,2,1,1,1,1>
      field_a = TestFunctionOffset(i, field_b(i - 1));               // EXPECTED_ACCESSES: W:field_a:<0,0,0,0,0,0> %and% R:field_b:<0,0,0,0,0,0>
      field_a = TestFunctionOffset(j, field_b(i - 1));               // EXPECTED_ACCESSES: W:field_a:<0,0,0,0,0,0> %and% R:field_b:<-1,-1,1,1,0,0>
      field_a = TestFunctionOffset(k, field_b(i - 1, k - 1));        // EXPECTED_ACCESSES: W:field_a:<0,0,0,0,0,0> %and% R:field_b:<-1,-1,0,0,0,0>
    }
  }
};

int main() {}
