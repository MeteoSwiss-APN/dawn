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

#define GT_VECTOR_LIMIT_SIZE 30

#include <gtest/gtest.h>
#include "test/integration-test/CodeGen/Macros.hpp"
#include "gridtools/clang/verify.hpp"
#include "test/integration-test/CodeGen/Options.hpp"
#include "test/integration-test/CodeGen/generated/kcache_fill_c++-naive.cpp"

#ifndef OPTBACKEND
#define OPTBACKEND gt
#endif

// clang-format off
#include INCLUDE_FILE(test/integration-test/CodeGen/generated/kcache_fill_,OPTBACKEND.cpp)
// clang-format on

using namespace dawn;
TEST(kcache_fill, test) {
  domain dom(Options::getInstance().m_size[0], Options::getInstance().m_size[1],
             Options::getInstance().m_size[2]);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);

  verifier verif(dom);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize() + 1);
  storage_t in(meta_data, "in"), out_gt(meta_data, "out-gt"), out_naive(meta_data, "out-naive");

  verif.fillMath(8.0, 2.0, 1.5, 1.5, 2.0, 4.0, in);
  verif.fill(-1.0, out_gt, out_naive);

  dawn_generated::OPTBACKEND::kcache_fill kcache_fill_gt(dom, in, out_gt);
  dawn_generated::cxxnaive::kcache_fill kcache_fill_naive(dom, in, out_naive);

  kcache_fill_gt.run();
  kcache_fill_naive.run();

  ASSERT_TRUE(verif.verify(out_gt, out_naive));
}
