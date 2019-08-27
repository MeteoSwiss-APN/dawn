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

#undef FUSION_MAX_VECTOR_SIZE
#undef FUSION_MAX_MAP_SIZE
#define FUSION_MAX_VECTOR_SIZE GT_VECTOR_LIMIT_SIZE
#define FUSION_MAX_MAP_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_LIMIT_VECTOR_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS

#include <gtest/gtest.h>
#include "test/integration-test/CodeGen/Macros.hpp"
#include "gridtools/clang/verify.hpp"
#include "test/integration-test/CodeGen/Options.hpp"
#include "test/integration-test/CodeGen/generated/kcache_flush_c++-naive.cpp"

#ifndef OPTBACKEND
#define OPTBACKEND gt
#endif

// clang-format off
#include INCLUDE_FILE(test/integration-test/CodeGen/generated/kcache_flush_,OPTBACKEND.cpp)
// clang-format on

using namespace dawn;
TEST(kcache_flush, test) {
  domain dom(Options::getInstance().m_size[0], Options::getInstance().m_size[1],
             Options::getInstance().m_size[2]);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);

  verifier verif(dom);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize() + 1);
  // Output fields
  storage_t out_opt(meta_data, "out_optimized"), out_naive(meta_data, "out_naive");

  // Input fields
  storage_t in(meta_data, "in");

  verif.fillMath(8.0, 2.0, 1.5, 1.5, 2.0, 4.0, in);
  verif.fill(-1.0, out_opt, out_naive);

  dawn_generated::OPTBACKEND::kcache_flush kcache_flush_gt(dom, in, out_opt);
  dawn_generated::cxxnaive::kcache_flush kcache_flush_naive(dom, in, out_naive);

  kcache_flush_gt.run();
  kcache_flush_naive.run();

  ASSERT_TRUE(verif.verify(out_opt, out_naive));
}
