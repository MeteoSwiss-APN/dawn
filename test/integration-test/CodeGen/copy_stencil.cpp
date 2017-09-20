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

// RUN: %gtclang% %file% -o%filename%_gen.cpp | %c++% %filename%_gen.cpp %gridtools_flags% -o%tmpdir%/%filename% | %tmpdir%/%filename%

#include "gridtools/clang_dsl.hpp"
#include "gridtools/clang/verify.hpp"

using namespace gridtools::clang;

stencil copy_stencil {
  storage in, out;

  Do {
    vertical_region(k_start, k_end)
      out = in;
  }
};

int main() {
  domain dom(64, 64, 80);
  verifier verif(dom);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize());
  storage_t in(meta_data, "in"), out(meta_data, "out");

  verif.for_each([&](int i, int j, int k) { return i + j + k; }, in);
  verif.fill(-1.0, out);

  copy_stencil copy(dom, in, out);
  copy.run();

  return !verif.verify(out, in);
}
