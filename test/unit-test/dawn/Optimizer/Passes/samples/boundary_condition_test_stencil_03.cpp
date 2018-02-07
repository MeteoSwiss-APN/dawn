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

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

globals { double glob_foo = 12; };

stencil_function zero {
  storage a;
  Do { a = 0; }
};

stencil SplitStencil {
  storage foo;
  storage bar;

  boundary_condition(zero(), foo);
  boundary_condition(zero(), bar);

  void Do() {
    vertical_region(k_start, k_end) {
      foo = bar[i + 1];
      bar = foo[i - 1] + glob_foo;
    }
  }
};
