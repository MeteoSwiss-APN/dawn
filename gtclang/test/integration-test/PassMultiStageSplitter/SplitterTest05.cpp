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

// RUN: %gtclang% %file% -fno-codegen -freport-pass-multi-stage-split

#include "gtclang_dsl_defs/gtclang_dsl.hpp"

using namespace gridtools::clang;

stencil Test {
  storage field_a, field_b, field_c, field_d;

  Do {
    vertical_region(k_start + 1, k_end - 1) {
      field_c = field_d; // EXPECTED: PASS: PassMultiStageSplitter: Test: split:%line% looporder:forward
      field_b = field_c(k + 1);
      field_a = field_b(k - 1);
    }
  }
};

int main() {}
