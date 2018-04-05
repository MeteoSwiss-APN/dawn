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
#include <gtest/gtest.h>
#include "test/integration-test/CodeGen/Options.hpp"
#include "gridtools/clang/verify.hpp"
#include "test/integration-test/CodeGen/generated/globals_stencil_gridtools.cpp"
#include "test/integration-test/CodeGen/generated/globals_stencil_c++-naive.cpp"

using namespace dawn;
TEST(globals_stencil, test) {
  domain dom(Options::getInstance().m_size[0], Options::getInstance().m_size[1], Options::getInstance().m_size[2]);
  verifier verif(dom);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize());
  storage_t in(meta_data, "in"), out_gt(meta_data, "out-gt"), out_naive(meta_data, "out-naive");

  verif.fillMath(8.0, 2.0, 1.5, 1.5, 2.0, 4.0, in);
  verif.fill(-1.0, out_gt, out_naive);

  gridtools::globals::get().var_runtime = 1;
  cxxnaive::globals::get().var_runtime = 1;

  gridtools::globals_stencil globals_gt(dom, in, out_gt);
  cxxnaive::globals_stencil globals_naive(dom, in, out_naive);

  globals_gt.run();
  verif.sync_storages(in);
  globals_naive.run();

  ASSERT_TRUE(verif.verify(out_gt, out_naive));
}
