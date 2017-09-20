//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _     _ _              _            _
//                        (_)   | | |            | |          | |
//               __ _ _ __ _  __| | |_ ___   ___ | |___    ___| | __ _ _ __   __ _
//              / _` | '__| |/ _` | __/ _ \ / _ \| / __|  / __| |/ _` | '_ \ / _` |
//             | (_| | |  | | (_| | || (_) | (_) | \__ \ | (__| | (_| | | | | (_| |
//              \__, |_|  |_|\__,_|\__\___/ \___/|_|___/  \___|_|\__,_|_| |_|\__, |
//               __/ |                                                        __/ |
//              |___/                                                        |___/
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

// RUN: %gtclang% %file% -fno-codegen -freport-accesses

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

stencil_function TestFunction01 {
  storage in;

  Do {
    return in(i + 1, j + 1, k + 1); // EXPECTED_ACCESSES: R:in:<0,1,0,1,0,1>
  }
};

stencil_function TestFunction02 {
  storage in;

  Do {
    return in(i + 1, j + 1); // EXPECTED_ACCESSES: R:in:<0,1,0,1,0,0>
  }
};

stencil Test {
  storage field_a, field_b;

  Do {
    vertical_region(k_start, k_end) {
      field_a = TestFunction01(field_b(i + 1, j + 1, k + 1)); // EXPECTED_ACCESSES: R:field_b:<0,2,0,2,0,2> %and% W:field_a:<0,0,0,0,0,0>
      field_a = TestFunction01(field_b(i + 1, j + 1, k));     // EXPECTED_ACCESSES: R:field_b:<0,2,0,2,0,1>

      field_a = TestFunction02(field_b(i + 1, j + 1, k + 1)); // EXPECTED_ACCESSES: R:field_b:<0,2,0,2,0,1>
      field_a = TestFunction02(field_b(i + 1, j + 1, k));     // EXPECTED_ACCESSES: R:field_b:<0,2,0,2,0,0>
    }
  }
};

int main() {}
