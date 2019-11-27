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

using namespace gridtools::clang;

stencil Test1 {
  storage in, out;

  Do {
    vertical_region(k_start, k_end) {
      in = out;
    }
  }
};

stencil Test2 {
  storage in, out;

  Do {
    Test1(in, out, in); // EXPECTED_ERROR: no matching stencil for call to 'Test1'
  }
};

int main() {}
