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

#include <gtest/gtest.h>
#include "test/integration-test/CodeGen/Options.hpp"
#include "gridtools/clang/verify.hpp"
#include "test/integration-test/CodeGen/generated/nested_stencil_functions_gridtools.cpp"
#include "test/integration-test/CodeGen/generated/nested_stencil_functions_c++-naive.cpp"

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
        out(i, j, k) = ((in(i + 2, j + 1, k) - in(i + 1, j + 1, k)) - (in(i + 2, j, k) - in(i + 1, j, k))) -
                       ((in(i + 1, j + 1, k) - in(i, j + 1, k)) - (in(i + 1, j, k) - in(i, j, k)));
      }
    }
  }
}
}
TEST(nested_stencil_functions, test_04) {
  domain dom(Options::getInstance().m_size[0], Options::getInstance().m_size[1], Options::getInstance().m_size[2]);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);

  verifier verif(dom);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize());
  storage_t in(meta_data, "in"), out_ref(meta_data, "out-ref"), out_naive(meta_data, "out-naive");

  verif.fillMath(8.0, 2.0, 1.5, 1.5, 2.0, 4.0, in);
  verif.fill(-1.0, out_ref, out_naive);

  cxxnaive::test_04_stencil test_04_naive(dom, in, out_naive);
  nsftest::test_04_stencil_reference(dom, in, out_ref);

  test_04_naive.run();

  ASSERT_TRUE(verif.verify(out_naive, out_ref));
}

TEST(nested_stencil_functions, test_05) {
  domain dom(Options::getInstance().m_size[0], Options::getInstance().m_size[1], Options::getInstance().m_size[2]);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);

  verifier verif(dom);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize());
  storage_t in(meta_data, "in"), out_ref(meta_data, "out-ref"), out_naive(meta_data, "out-naive");

  verif.fillMath(8.0, 2.0, 1.5, 1.5, 2.0, 4.0, in);
  verif.fill(-1.0, out_ref, out_naive);

  cxxnaive::test_05_stencil test_05_naive(dom, in, out_naive);
  nsftest::test_05_stencil_reference(dom, in, out_ref);

  test_05_naive.run();

  ASSERT_TRUE(verif.verify(out_naive, out_ref));
}
