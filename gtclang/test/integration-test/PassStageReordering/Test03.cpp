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

// RUN: %gtclang% %file% -fno-codegen -freport-pass-stage-reordering
// EXPECTED_FILE: OUTPUT:%filename%_before.json,%filename%_after.json REFERENCE:%filename%_before_ref.json,%filename%_after_ref.json

#include "gtclang_dsl_defs/gtclang_dsl.hpp"

using namespace gtclang::dsl;

stencil Test {
  storage field_b0, field_b1, field_b2;

  Do {
    vertical_region(k_end - 1, k_start + 1) {
      field_b1 = field_b0;
      field_b2 = field_b1(k - 1);
    }
  }
};

int main() {}
