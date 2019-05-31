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

stencil_function TestFunction {
  storage field_a;

  Do {
    return field_a(i + 1);
  }
};

stencil Test {
  storage field_a;

  Do {
    vertical_region(k_start, k_end) {
      field_a = TestFunction(field_a); // EXPECTED: PASS: PassFieldVersioning: Test: rename:%line% field_a:field_a_0
    }
  }
};

int main() {}
