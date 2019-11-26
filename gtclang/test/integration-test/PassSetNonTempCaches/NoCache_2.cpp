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
// EXPECTED: PASS: PassSetNonTempCaches: NotEnoughReadsWithReadBeforeWrite : no fields cached
#include "gtclang_dsl_defs/gtclang_dsl.hpp"

using namespace gtclang::dsl;
using namespace gridtools;

stencil NotEnoughReadsWithReadBeforeWrite {
  storage field_a, field_b, field_c, field_d, field_e;

  void Do() {
    vertical_region(k_start, k_end - 1) {
      field_b = field_a;
      field_a[j + 1] = field_b[k + 1];
    }
  }
};
