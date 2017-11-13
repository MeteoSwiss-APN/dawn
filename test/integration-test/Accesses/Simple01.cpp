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

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

stencil Test {
  storage field_a, field_b;

  Do {
    vertical_region(k_start, k_end)
        field_a = field_b; // EXPECTED_ACCESSES: W:field_a:<0,0,0,0,0,0> %and% R:field_b:<0,0,0,0,0,0>
  }
};

int main() {}
