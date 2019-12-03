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

// RUN: %gtclang% %file% -fno-codegen -freport-accesses

#include "gtclang_dsl_defs/gtclang_dsl.hpp"

using namespace gtclang::dsl;

stencil Test {
  storage field_a, field_b;

  Do {
    vertical_region(k_start, k_end)
        field_b = field_a(i + 1, j + 1) > 0.0 ? field_a(i - 1) : field_a(j - 1); // EXPECTED_ACCESSES: W:field_b:<[<no_horizontal_extent>,(0,0)]> %and% R:field_a:<[(-1,1),(-1,1),(0,0)]>
  }
};

int main() {}
