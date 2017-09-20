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
// EXPECTED: PASS: PassSetCaches: Test: MS0: tmp:K:epflush
// EXPECTED: PASS: PassSetCaches: Test: MS1: tmp:K:bpfill

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

stencil Test {
  storage a, b, c;
  temporary_storage tmp;

  Do {
    // MS0
    vertical_region(k_start, k_start)
      tmp = a;

    vertical_region(k_start+1, k_end)
      b = tmp(k-1);

    // MS1
    vertical_region(k_end, k_end)
      tmp = (b(k-1) + b) * tmp;
    
    vertical_region(k_end-1, k_start)
      c = tmp(k+1);

  }
};

int main() {}
