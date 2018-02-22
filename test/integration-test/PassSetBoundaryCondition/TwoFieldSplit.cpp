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
// RUN: %gtclang% %file% -fno-codegen -fsplit-stencils -freport-bc -max-fields=2
// EXPECTED: PASS: PassSetBoundaryCondition: Test01 : Boundary Condition for field 'foo' inserted
#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

globals {
  double glob_bar;
  double glob_foo;
};

stencil_function zero {
  storage a;
  Do { a = 0; }
};

stencil Test01 {
  storage foo;
  storage bar;
  storage fooo;
  storage barr, test;

  boundary_condition(zero(), foo);

  void Do() {
    vertical_region(k_start, k_end) {
      foo = bar[i + 1];
      bar = foo[i - 1] + glob_foo;
      // mytest(fooo, barr, zero(foo, bar));
    }
  }
};
