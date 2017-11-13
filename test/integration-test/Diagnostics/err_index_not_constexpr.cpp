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

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

stencil Test {
  storage in, out;

  Do {
    vertical_region(k_start, k_end) {
      int kk = 5;
      in = out(i, j, k + kk); // EXPECTED_ERROR: offset of index is not a constant expression
    }
  }
};

int main() {}
