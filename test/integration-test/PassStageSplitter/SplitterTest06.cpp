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

// RUN: %gtclang% %file% -fno-codegen -freport-pass-stage-split

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

stencil Test {
  storage field_a, field_b, field_c, field_d, field_e, field_f, field_g, field_h, field_i;

  Do {
    vertical_region(k_start, k_end) {
      field_h = field_i; // EXPECTED: PASS: PassStageSplitter: Test: split:%line%
      field_d = field_e;
      field_c = field_d;
      field_b = field_f(i + 1) + field_d;
      field_g = field_h(i + 1);
      field_a = field_b + field_c;
    }
  }
};

int main() {}
