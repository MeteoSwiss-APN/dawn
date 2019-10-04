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

// =================================================================================================
//
//          This test is currently disabled due to an error in the gridtools library
//
//            Gridtools nested function calls are currently creating wrong results and are
//            threfore not supported currenlty. This led to the most recent update breaking them and
//            until fixed, these tests are disabled.
//            We currently do not intend to move away from our usage of nested functions but this
//            will be reevaluated if gridtools choses to abandon nested function calls
// =================================================================================================

#define GRIDTOOLS_CLANG_GENERATED 1
#define GRIDTOOLS_CLANG_HALO_EXTEND 3
#define GT_VECTOR_LIMIT_SIZE 30

#undef FUSION_MAX_VECTOR_SIZE
#undef FUSION_MAX_MAP_SIZE
#define FUSION_MAX_VECTOR_SIZE GT_VECTOR_LIMIT_SIZE
#define FUSION_MAX_MAP_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_LIMIT_VECTOR_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS

#include "gridtools/clang/verify.hpp"
#include "gridtools/clang_dsl.hpp"
#include "test/integration-test/CodeGen/Options.hpp"
#include <gtest/gtest.h>
//#include "test/integration-test/CodeGen/generated/nested_stencil_functions_gridtools.cpp"
//#include "test/integration-test/CodeGen/generated/nested_stencil_functions_c++-naive.cpp"

using namespace gridtools::clang;
using namespace dawn;

namespace nsftest {
void test_04_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  for(int i = dom.iminus(); i < (dom.isize() - dom.iplus()); ++i) {
    for(int j = dom.jminus(); j < (dom.jsize() - dom.jplus()); ++j) {
      for(int k = dom.kminus(); k < (dom.ksize() - dom.kplus()); ++k) {
        out(i, j, k) = (in(i + 1, j, k) - in(i, j, k)) + (in(i, j + 1, k) - in(i, j, k));
      }
    }
  }
}
void test_05_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  for(int i = dom.iminus(); i < (dom.isize() - dom.iplus()); ++i) {
    for(int j = dom.jminus(); j < (dom.jsize() - dom.jplus()); ++j) {
      for(int k = dom.kminus(); k < (dom.ksize() - dom.kplus()); ++k) {
        out(i, j, k) =
            ((in(i + 2, j + 1, k) - in(i + 1, j + 1, k)) - (in(i + 2, j, k) - in(i + 1, j, k))) -
            ((in(i + 1, j + 1, k) - in(i, j + 1, k)) - (in(i + 1, j, k) - in(i, j, k)));
      }
    }
  }
}
}

TEST(nested_stencil_functions, test_04) {
  domain dom(Options::getInstance().m_size[0], Options::getInstance().m_size[1],
             Options::getInstance().m_size[2]);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);

  verifier verif(dom);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize());
  storage_t in(meta_data, "in"), out_ref(meta_data, "out-ref"), out_naive(meta_data, "out-naive");

  verif.fillMath(8.0, 2.0, 1.5, 1.5, 2.0, 4.0, in);
  verif.fill(-1.0, out_ref, out_naive);

  //  dawn_generated::cxxnaive::test_04_stencil test_04_naive(dom, in, out_naive);
  //  nsftest::test_04_stencil_reference(dom, in, out_ref);

  //  test_04_naive.run();

  ASSERT_TRUE(verif.verify(out_naive, out_ref));
}

TEST(nested_stencil_functions, test_05) {
  domain dom(Options::getInstance().m_size[0], Options::getInstance().m_size[1],
             Options::getInstance().m_size[2]);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);

  verifier verif(dom);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize());
  storage_t in(meta_data, "in"), out_ref(meta_data, "out-ref"), out_naive(meta_data, "out-naive");

  verif.fillMath(8.0, 2.0, 1.5, 1.5, 2.0, 4.0, in);
  verif.fill(-1.0, out_ref, out_naive);

  //  dawn_generated::cxxnaive::test_05_stencil test_05_naive(dom, in, out_naive);
  //  nsftest::test_05_stencil_reference(dom, in, out_ref);

  //  test_05_naive.run();

  ASSERT_TRUE(verif.verify(out_naive, out_ref));
}
