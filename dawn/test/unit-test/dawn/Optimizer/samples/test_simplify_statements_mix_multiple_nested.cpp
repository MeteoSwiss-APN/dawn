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

// gtclang test_simplify_statements_mix_multiple_nested.cpp -fno-codegen -fwrite-iir -fkeep-varnames

#include "gtclang_dsl_defs/gtclang_dsl.hpp"

using namespace gtclang::dsl;

stencil stencil {
  storage a, d;

  Do {
    vertical_region(k_start, k_end) { 
      int b = d;
      int c = d;
      a += ++b + (1 + --c);
      a *= ++c * --b;
    }
  }
};
