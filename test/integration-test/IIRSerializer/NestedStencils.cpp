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

// RUN: %gtclang% %file% -fwrite-iir -fno-codegen -o %filename%_gen.cpp
// EXPECTED_FILE: OUTPUT:%filename%.2.iir REFERENCE:%filename%_ref.iir IGNORE:filename DELETE:%filename%.0.iir,%filename%.1.iir

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

stencil Test {
  storage field_a, field_b;

  Do {
    vertical_region(k_start, k_end) { field_a = field_b; }
  }
};

stencil Nesting1 {
  storage filed_c, field_d;

  Do { Test(filed_c, field_d); }
};

stencil Nesting2 {
  storage field_e, field_f;

  Do {
    Nesting1(field_e, field_f);
    Test(field_f, field_e);
  }
};

