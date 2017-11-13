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

// RUN: %gtclang% %file% -o%filename%_gen.cpp | %c++% %filename%_gen.cpp %gridtools_flags% -o%tmpdir%/%filename%

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

interval k_flat = k_start + 11;

// Check if we correclty generate the empty Do-Methods according to
// https://github.com/eth-cscs/gridtools/issues/330

stencil EmptyDoMethodTest {
  storage foo, bar;

  Do {
    // Should give empty Do-Method for [k_flat+1, k_end]
    vertical_region(k_start, k_flat)
        foo = bar;

    // Should give empty Do-Method for [k_start, k_flat-1], [k_flat+1, k_end]
    vertical_region(k_flat, k_flat)
        foo = bar;

    // Should give empty Do-Method for [k_start, k_flat-1]
    vertical_region(k_flat, k_end)
        foo = bar;
  }
};

int main() {}
