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

// RUN: %gtclang% %file% -o%filename%_gen.cpp --config=%filedir%/globals.json | %c++% %filename%_gen.cpp %gridtools_flags% -o%tmpdir%/%filename% | %tmpdir%/%filename%

#include "gridtools/clang_dsl.hpp"
#include "gridtools/clang/verify.hpp"

using namespace gridtools::clang;

globals {
  int var_runtime;         // == 1
  double var_default = 2;  // == 2
  bool var_compiletime;    // == true
};

stencil Test01 {
  storage s;

  Do {
    vertical_region(k_start, k_end) {
      if(var_compiletime) // true
        s = var_runtime + var_default; // 1 + 2
    }
  }
};

int main() {
  domain dom(64, 64, 80);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);
  verifier verif(dom);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize());
  storage_t s(meta_data, "s"), s_ref(meta_data, "s_ref");

  verif.fill(-1.0, s);
  verif.fill(3.0, s_ref);

  globals::get().var_runtime = 1;

  Test01 test(dom, s);
  test.run();

  return !verif.verify(s, s_ref);
}

