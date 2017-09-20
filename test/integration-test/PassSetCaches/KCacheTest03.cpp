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

// RUN: %gtclang% %file% -fno-codegen -fuse-kcaches -freport-pass-set-caches
// EXPECTED: PASS: PassSetCaches: Test: MS0: tmp:K:flush
// EXPECTED: PASS: PassSetCaches: Test: MS1: tmp:K:fill

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

stencil Test {
  storage in, out;
  storage a, b, c;
  temporary_storage tmp;

  Do {
    vertical_region(k_start, k_end) {
      // --- MS0 ---
      tmp = in;
      
      b = a;
      
      // --- MS1 ---
      c = b(k+1);
      c = b(k-1);
      
      out = tmp;
    }
  }
};

int main() {}
