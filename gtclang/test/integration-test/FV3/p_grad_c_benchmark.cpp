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
#include "gtclang_dsl_defs/verify.hpp"
#include "test/integration-test/CodeGen/Macros.hpp"
#include "test/integration-test/CodeGen/Options.hpp"
#include "test/integration-test/FV3/generated/p_grad_c_c++-naive.cpp"

#ifndef OPTBACKEND
#define OPTBACKEND gt
#endif

// clang-format off
#include INCLUDE_FILE(test/integration-test/FV3/generated/p_grad_c_,OPTBACKEND.cpp)
// clang-format on

using namespace dawn;
TEST(p_grad_c, test) {
  domain dom(Options::getInstance().m_size[0], Options::getInstance().m_size[1],
             Options::getInstance().m_size[2]);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);
  verifier verif(dom);

  meta_data_ij_t meta_data_ij(dom.isize(), dom.jsize(), 1);
  storage_ij_t delpc(meta_data_ij, "delpc");

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize() + 1);
  storage_t pkc(meta_data, "pkc");
  storage_t gz(meta_data, "gz");
  storage_t uc_gt(meta_data, "uc_gt"), uc_cxxnaive(meta_data, "uc_cxxnaive");
  storage_t vc_gt(meta_data, "vc_gt"), vc_cxxnaive(meta_data, "vc_cxxnaive");
  storage_t rdxc(meta_data, "rdxc");
  storage_t rdyc(meta_data, "rdyc");

  dawn_generated::OPTBACKEND::p_grad_c p_grad_c_gt(dom);
  dawn_generated::cxxnaive::p_grad_c p_grad_c_cxxnaive(dom);

  verif.fill(-1.0, delpc);
  verif.fillMath(5.0, 1.2, 1.3, 1.7, 2.2, 3.5, pkc);
  verif.fillMath(5.0, 1.2, 1.3, 1.7, 2.2, 3.5, gz, rdxc, rdyc);

  double dt2 = 0.001;
  p_grad_c_gt.set_dt2(dt2);
  p_grad_c_cxxnaive.set_dt2(dt2);

  // Test with hydrostatic = true
  // {
  verif.fillMath(8.0, 2.0, 1.5, 1.5, 2.0, 4.0, uc_gt, vc_gt, uc_cxxnaive, vc_cxxnaive);

  p_grad_c_gt.set_hydrostatic(true);
  p_grad_c_cxxnaive.set_hydrostatic(true);

  p_grad_c_gt.run(delpc, pkc, gz, uc_gt, vc_gt, rdxc, rdyc);
  p_grad_c_cxxnaive.run(delpc, pkc, gz, uc_cxxnaive, vc_cxxnaive, rdxc, rdyc);

  ASSERT_TRUE(verif.verify(uc_gt, uc_cxxnaive));
  ASSERT_TRUE(verif.verify(vc_gt, vc_cxxnaive));
  // }

  // Test with hydrostatic = false
  // {
  verif.fillMath(8.0, 2.0, 1.5, 1.5, 2.0, 4.0, uc_gt, vc_gt, uc_cxxnaive, vc_cxxnaive);

  p_grad_c_gt.set_hydrostatic(false);
  p_grad_c_cxxnaive.set_hydrostatic(false);

  p_grad_c_gt.run(delpc, pkc, gz, uc_gt, vc_gt, rdxc, rdyc);
  p_grad_c_cxxnaive.run(delpc, pkc, gz, uc_cxxnaive, vc_cxxnaive, rdxc, rdyc);

  ASSERT_TRUE(verif.verify(uc_gt, uc_cxxnaive));
  ASSERT_TRUE(verif.verify(vc_gt, vc_cxxnaive));
  // }
}
