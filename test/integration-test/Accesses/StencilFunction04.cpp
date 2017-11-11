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
    return in(dir + 1);
  }
};

stencil_function bar {
  direction dir;
  storage in1;
  storage in2;

  Do {
    return in1(dir + 1) + in2(dir + 1);
  }
};

stencil_function baz {
  direction dir;
  storage in1;
  storage in2;

  Do {
    double ret = in1(dir + 1) + in2(dir + 1);
    return ret;
  }
};

stencil Test {
  storage field_a, field_b, field_c;

  Do {
    vertical_region(k_start, k_end) {
      field_a = foo(i, field_b);                                         // EXPECTED_ACCESSES: R:field_b:<0,1,0,0,0,0>
      field_a = foo(j, foo(i, field_b));                                 // EXPECTED_ACCESSES: R:field_b:<0,1,0,1,0,0>
      field_a = foo(k, foo(j, foo(i, field_b)));                         // EXPECTED_ACCESSES: R:field_b:<0,1,0,1,0,1>
      field_a = foo(k, foo(j, foo(i, foo(k, foo(j, foo(i, field_b)))))); // EXPECTED_ACCESSES: R:field_b:<0,2,0,2,0,2>

      field_a = bar(i, field_b, field_c);                  // EXPECTED_ACCESSES: R:field_b:<0,1,0,0,0,0> %and% R:field_c:<0,1,0,0,0,0>
      field_a = bar(j, foo(i, field_b), field_c);          // EXPECTED_ACCESSES: R:field_b:<0,1,0,1,0,0> %and% R:field_c:<0,0,0,1,0,0>
      field_a = bar(j, bar(i, field_b, field_c), field_c); // EXPECTED_ACCESSES: R:field_b:<0,1,0,1,0,0> %and% R:field_c:<0,1,0,1,0,0>
      field_a = bar(i, bar(i, field_b, field_c), field_c); // EXPECTED_ACCESSES: R:field_b:<0,2,0,0,0,0> %and% R:field_c:<0,2,0,0,0,0>

      field_a = baz(i, field_b, field_c);                  // EXPECTED_ACCESSES: R:field_b:<0,1,0,0,0,0> %and% R:field_c:<0,1,0,0,0,0>
      field_a = baz(j, foo(i, field_b), field_c);          // EXPECTED_ACCESSES: R:field_b:<0,1,0,1,0,0> %and% R:field_c:<0,0,0,1,0,0>
      field_a = baz(j, baz(i, field_b, field_c), field_c); // EXPECTED_ACCESSES: R:field_b:<0,1,0,1,0,0> %and% R:field_c:<0,1,0,1,0,0>
      field_a = baz(i, baz(i, field_b, field_c), field_c); // EXPECTED_ACCESSES: R:field_b:<0,2,0,0,0,0> %and% R:field_c:<0,2,0,0,0,0>
    }
  }
};

int main() {}
