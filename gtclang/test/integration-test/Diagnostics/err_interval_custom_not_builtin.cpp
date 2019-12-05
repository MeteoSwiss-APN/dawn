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

// RUN: %gtclang% %file% -fno-codegen

#include "gtclang_dsl_defs/gtclang_dsl.hpp"

using namespace gtclang::dsl;

interval k_flat = k_start + 11;
interval k_flat2 = k_flat + 1; // EXPECTED_ERROR: invalid built-in interval 'k_flat'

stencil Test {
  storage a;

  Do {
    vertical_region(k_flat, k_end)
        a = 0.0;

    vertical_region(k_flat2, k_end)
        a = 0.0;
  }
};

int main() {}
