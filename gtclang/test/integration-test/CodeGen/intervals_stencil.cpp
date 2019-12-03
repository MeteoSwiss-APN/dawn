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

#include "gtclang_dsl_defs/gtclang_dsl.hpp"

using namespace gtclang::dsl;

#ifndef DAWN_GENERATED
interval k_flat = k_start + 4;
#endif

// Check if we correclty generate the empty Do-Methods according to
// https://github.com/eth-cscs/gridtools/issues/330

stencil intervals_stencil {
  storage in, out;

  Do {
    // Should give empty Do-Method for [k_flat+1, k_end]
    vertical_region(k_start, k_flat)
        out = in + 1;

    // Should give empty Do-Method for [k_start, k_flat-1], [k_flat+1, k_end]
    vertical_region(k_flat + 1, k_flat + 1)
        out = in + 2;

    // Should give empty Do-Method for [k_start, k_flat-1]
    vertical_region(k_flat + 2, k_end)
        out = in + 3;
  }
};
