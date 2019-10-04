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

// RUN: %gtclang% %file% -fno-codegen -fmerge-temporaries -freport-pass-temporary-merger
// EXPECTED: PASS: PassTemporaryMerger: Test: merging: tmp_a, tmp_b

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

stencil Test {
  storage field_a, field_b;
  var tmp_a, tmp_b;

  Do {
    vertical_region(k_start, k_end) {
      tmp_a = field_a;
      field_a = tmp_a(i + 1);

      tmp_b = field_b;
      field_b = tmp_b(i + 1);
    }
  }
};

int main() {}
