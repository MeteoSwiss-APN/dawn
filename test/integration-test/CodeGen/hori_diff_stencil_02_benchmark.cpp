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
#define GRIDTOOLS_CLANG_GENERATED 1
#define GRIDTOOLS_CLANG_HALO_EXTEND 3

#include "gridtools/clang/verify.hpp"
#include "test/integration-test/CodeGen/Options.hpp"
#include "test/integration-test/CodeGen/generated/hori_diff_stencil_02_c++-naive.cpp"
#include "test/integration-test/CodeGen/generated/hori_diff_stencil_02_gridtools.cpp"
#include <gtest/gtest.h>

using namespace dawn;
TEST(hori_diff_stencil_02, test) {
  domain dom(Options::getInstance().m_size[0], Options::getInstance().m_size[1],
             Options::getInstance().m_size[2]);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);

  verifier verif(dom);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize() + 1);
  storage_t u(meta_data, "u"), out_gt(meta_data, "out-gt"), out_naive(meta_data, "out-naive");

  verif.fillMath(8.0, 2.0, 1.5, 1.5, 2.0, 4.0, u);
  verif.fill(-1.0, out_gt, out_naive);

  gridtools::hori_diff_stencil hori_diff_gt(dom, u, out_gt);
  cxxnaive::hori_diff_stencil hori_diff_naive(dom, u, out_naive);

  hori_diff_gt.run();
  hori_diff_naive.run();

  ASSERT_TRUE(verif.verify(out_gt, out_naive));
}
