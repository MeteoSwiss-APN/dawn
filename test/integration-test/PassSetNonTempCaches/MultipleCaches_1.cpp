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

// RUN: %gtclang% %file% -fno-codegen -fcache-non-temp-fields -freport-cache-non-temp-fields
// EXPECTED: PASS: PassSetNonTempCaches: Test01 : Cached: field_b : Type: IJ:local Cached: field_c : Type: IJ:local
#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;
using namespace gridtools;

stencil Test01 {
  storage field_a, field_b, field_c, field_d, field_e;

  void Do() {
    vertical_region(k_start, k_start) {
      field_b[i + 1] = field_a;
      field_a = field_b[j - 1];
      field_c[i + 1] = field_d;
      field_c[j + 1] = field_a;
      field_c[j + 2] = field_b;
      field_c[i + 2] = field_b;
      field_c[i + 1] = field_b;
    }
  }
};
