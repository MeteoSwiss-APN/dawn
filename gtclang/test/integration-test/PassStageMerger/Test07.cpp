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
// clang-format off
// RUN: %gtclang% %file% -fno-codegen -fmerge-stages -fmerge-do-methods -freport-pass-stage-merger
// EXPECTED_FILE: OUTPUT:%filename%_before.json,%filename%_after.json REFERENCE:%filename%_before_ref.json,%filename%_after_ref.json
// clang-format on
#include "gtclang_dsl_defs/gtclang_dsl.hpp"

using namespace gtclang::dsl;

stencil Test {
  storage in, out;

  Do {

    vertical_region(k_start, k_end) { out = in; }
    vertical_region(k_start, k_end) { out = 0; }
  }
};
