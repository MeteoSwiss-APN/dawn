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
#include "test/integration-test/CodeGen/generated/coriolis_stencil_c++-naive.cpp"

#ifndef OPTBACKEND
#define OPTBACKEND gt
#endif

// clang-format off
#include INCLUDE_FILE(test/integration-test/CodeGen/generated/coriolis_stencil_,OPTBACKEND.cpp)
// clang-format on

using namespace dawn;
TEST(coriolis_stencil, test) {
  domain dom(Options::getInstance().m_size[0], Options::getInstance().m_size[1],
             Options::getInstance().m_size[2]);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);
  verifier verif(dom);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize() + 1);
  storage_t u_tens_gt(meta_data, "u_tens_gt"), u_tens_cxxnaive(meta_data, "u_tens_cxxnaive");
  storage_t v_tens_gt(meta_data, "v_tens_gt"), v_tens_cxxnaive(meta_data, "v_tens_cxxnaive");
  storage_t u_nnow(meta_data, "u_nnow"), v_nnow(meta_data, "v_nnow"), fc(meta_data, "fc");

  verif.fill(-1.0, u_tens_gt, v_tens_gt, u_tens_cxxnaive, v_tens_cxxnaive);
  verif.fillMath(8.0, 2.0, 1.5, 1.5, 2.0, 4.0, u_nnow);
  verif.fillMath(5.0, 1.2, 1.3, 1.7, 2.2, 3.5, v_nnow);
  verif.fillMath(2.0, 1.3, 1.4, 1.6, 2.1, 3.0, fc);

  dawn_generated::OPTBACKEND::coriolis_stencil coriolis_gt(dom, u_tens_gt, u_nnow, v_tens_gt, v_nnow, fc);
  dawn_generated::cxxnaive::coriolis_stencil coriolis_cxxnaive(dom, u_tens_cxxnaive, u_nnow, v_tens_cxxnaive,
                                               v_nnow, fc);

  coriolis_gt.run();
  coriolis_cxxnaive.run();

  ASSERT_TRUE(verif.verify(u_tens_gt, u_tens_cxxnaive));
  ASSERT_TRUE(verif.verify(v_tens_gt, v_tens_cxxnaive));
}
