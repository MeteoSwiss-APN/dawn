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

// RUN: %gtclang% %file% -fno-codegen -freport-pass-field-versioning

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

stencil Test {
  storage field, tmp;

  Do {
    vertical_region(k_start, k_end) {
      tmp = field(i + 1); // EXPECTED: PASS: PassFieldVersioning: Test: rename:%line% field_0:field_1
      field = tmp;        // EXPECTED: PASS: PassFieldVersioning: Test: rename:%line% tmp:tmp_0

      tmp = field(i + 1); // EXPECTED: PASS: PassFieldVersioning: Test: rename:%line% field:field_0
      field = tmp;
    }
  }
};

int main() {}
