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

struct foo {
  operator int() { return true; }
};

globals {
  int a = foo(); // EXPECTED_ERROR: invalid type 'int' of default value of global variable 'a'
};

stencil Test {
  storage in, out;

  Do {
    vertical_region(k_start, k_end)
        in = out;
  }
};

int main() {}
