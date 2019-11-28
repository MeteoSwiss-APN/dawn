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

// RUN: %gtclang% %file% -fno-codegen -freport-pass-temporary-type
// EXPECTED: PASS: PassTemporaryType: Test: promote:.*local_variable.*

#include "gtclang_dsl_defs/gtclang_dsl.hpp"

using namespace gridtools::clang;

stencil Test {
  storage field_a, field_b, field_c;

  Do {
    vertical_region(k_start, k_end) {
      double local_variable = 5.0;

      local_variable *= local_variable;

      field_a = field_b;
      field_c = field_a(i + 1) + local_variable * local_variable;

      field_a = field_b;
      field_c = field_a(i + 1) + local_variable * local_variable;
    }
  }
};

int main() {}
