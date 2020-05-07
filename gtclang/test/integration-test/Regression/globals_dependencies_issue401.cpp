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

// RUN: %gtclang% %file% -fno-codegen -write-iir -o %filename%.cpp
// EXPECTED_FILE: OUTPUT:copystencil.iir REFERENCE:%filename%_ref.0.iir IGNORE:filename

#include "gtclang_dsl_defs/gtclang_dsl.hpp"

using namespace gtclang::dsl;

globals { int global = 0; };

stencil copystencil {
  storage in_field, out_field;
  void Do() {
    if(global == 0) { // nesting problem, issue #261
      vertical_region(k_start, k_end) { out_field = global * in_field; }
      global = 1;
      vertical_region(k_start, k_end) { out_field = global * in_field; }
    }
  }
};
