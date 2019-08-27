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
#include "test/integration-test/CodeGen/generated/hd_smagorinsky_c++-naive.cpp"

#ifndef OPTBACKEND
#define OPTBACKEND gt
#endif

// clang-format off
#include INCLUDE_FILE(test/integration-test/CodeGen/generated/hd_smagorinsky_,OPTBACKEND.cpp)
// clang-format on

using namespace dawn;
TEST(hd_smagorinsky, test) {

  domain dom(Options::getInstance().m_size[0], Options::getInstance().m_size[1],
             Options::getInstance().m_size[2]);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);

  verifier verif(dom);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize() + 1);
  meta_data_j_t meta_data_j(1, dom.jsize(), 1);
  meta_data_scalar_t meta_data_scalar(1, 1, 1);

  // Output fields
  storage_t u_out_gt(meta_data, "v_out");
  storage_t v_out_gt(meta_data, "u_out");
  storage_t u_out_naive(meta_data, "v_out");
  storage_t v_out_naive(meta_data, "u_out");

  // Input fields
  storage_t u_in(meta_data, "u_in");
  storage_t v_in(meta_data, "v_in");
  storage_t hdmaskvel(meta_data, "hdmaskvel");
  storage_j_t crlavo(meta_data_j, "crlavo");
  storage_j_t crlavu(meta_data_j, "crlavu");
  storage_j_t crlato(meta_data_j, "crlato");
  storage_j_t crlatu(meta_data_j, "crlatu");
  storage_j_t acrlat0(meta_data_j, "acrlat0");

  // Scalar fields
  storage_t eddlon(meta_data, "eddlon");
  storage_t eddlat(meta_data, "eddlat");
  storage_t tau_smag(meta_data, "tau_smag");
  storage_t weight_smag(meta_data, "weight_smag");

  verif.fillMath(8.0, 2.0, 1.5, 1.5, 2.0, 4.0, u_in);
  verif.fillMath(6.0, 1.0, 0.9, 1.1, 2.0, 4.0, v_in);
  verif.fillMath(5.0, 2.2, 1.7, 1.9, 2.0, 4.0, hdmaskvel);
  verif.fillMath(6.5, 1.2, 1.7, 1.9, 2.1, 2.0, crlavo);
  verif.fillMath(5.0, 2.2, 1.7, 1.9, 2.0, 1.0, crlavu);
  verif.fillMath(6.5, 1.2, 1.7, 0.9, 2.1, 2.0, crlato);
  verif.fillMath(5.0, 2.2, 1.7, 0.9, 2.0, 1.0, crlatu);
  verif.fillMath(6.5, 1.2, 1.2, 1.2, 2.2, 2.2, acrlat0);

  verif.fill(-1.0, u_out_gt, v_out_gt, u_out_naive, v_out_naive);

  // Assemble the stencil ...
  dawn_generated::OPTBACKEND::hd_smagorinsky_stencil hd_smagorinsky_gt(
      dom, u_out_gt, v_out_gt, u_in, v_in, hdmaskvel, crlavo, crlavu, crlato, crlatu, acrlat0,
      eddlon, eddlat, tau_smag, weight_smag);
  dawn_generated::cxxnaive::hd_smagorinsky_stencil hd_smagorinsky_naive(
      dom, u_out_naive, v_out_naive, u_in, v_in, hdmaskvel, crlavo, crlavu, crlato, crlatu, acrlat0,
      eddlon, eddlat, tau_smag, weight_smag);

  hd_smagorinsky_gt.run();
  hd_smagorinsky_naive.run();

  ASSERT_TRUE(verif.verify(u_out_gt, u_out_naive));
  ASSERT_TRUE(verif.verify(v_out_gt, v_out_naive));
}
