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

// RUN: %gtclang% %file% -o%filename%_gen.cpp | %c++% %filename%_gen.cpp %gridtools_flags% -o%tmpdir%/%filename% -fsyntax-only

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

stencil_function laplacian {
  storage phi;

  Do { return phi(i + 1) + phi(i - 1) + phi(j + 1) + phi(j - 1) - 4.0 * phi; }
};

stencil hori_diff_stencil {
  storage u, out, lap;

  Do {
    vertical_region(k_start, k_end) {
      lap = laplacian(u);
      out = laplacian(lap);
    }
  }
};

int main() {
  domain dom(256, 256, 80);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);

  meta_data_t meta_data(256, 256, 80);
  storage_t u(meta_data, "u"), out(meta_data, "out"), lap(meta_data, "lap");

  hori_diff_stencil hori_diff(dom, u, out, lap);
}
