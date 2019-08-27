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
#include "gridtools/clang/verify.hpp"
#include "test/integration-test/CodeGen/Macros.hpp"
#include "test/integration-test/CodeGen/Options.hpp"
#include "test/integration-test/CodeGen/generated/hori_diff_type2_stencil_c++-naive.cpp"

#ifndef OPTBACKEND
#define OPTBACKEND gt
#endif

// clang-format off
#include INCLUDE_FILE(test/integration-test/CodeGen/generated/hori_diff_type2_stencil_,OPTBACKEND.cpp)
// clang-format on

using namespace dawn;
TEST(hori_diff_type2_stencil, test) {

  domain dom(Options::getInstance().m_size[0], Options::getInstance().m_size[1],
             Options::getInstance().m_size[2]);
  dom.set_halos(GRIDTOOLS_CLANG_HALO_EXTEND, GRIDTOOLS_CLANG_HALO_EXTEND,
                GRIDTOOLS_CLANG_HALO_EXTEND, GRIDTOOLS_CLANG_HALO_EXTEND, 0, 0);

  verifier verif(dom);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize() + 1);
  meta_data_j_t meta_data_j(1, dom.jsize(), 1);

  storage_t u(meta_data, "u"), hdmask(meta_data, "hdmask"), out_naive(meta_data, "out-naive"),
      u_out_gt(meta_data, "u"), u_out_naive(meta_data, "u");

  storage_j_t crlato(meta_data_j, "crlato"), crlatu(meta_data_j, "crlatu");

  verif.fillMath(8.0, 2.0, 1.5, 1.5, 2.0, 4.0, u);
  verif.fillMath(5.0, 2.2, 1.7, 1.9, 2.0, 4.0, hdmask);
  verif.fillMath(6.5, 1.2, 1.7, 0.9, 2.1, 2.0, crlato);
  verif.fillMath(5.0, 2.2, 1.7, 0.9, 2.0, 1.0, crlatu);

  verif.fill(-1.0, u_out_gt, u_out_naive);

  dawn_generated::OPTBACKEND::hori_diff_type2_stencil hd_gt(dom, u_out_gt, u, crlato, crlatu, hdmask);
  dawn_generated::cxxnaive::hori_diff_type2_stencil hd_naive(dom, u_out_naive, u, crlato, crlatu, hdmask);

  hd_gt.run();
  hd_naive.run();

  ASSERT_TRUE(verif.verify(u_out_gt, u_out_naive));
}
