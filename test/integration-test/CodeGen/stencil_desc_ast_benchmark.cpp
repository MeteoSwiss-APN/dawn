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
#include "test/integration-test/CodeGen/generated/stencil_desc_ast_gridtools.cpp"
#include "test/integration-test/CodeGen/generated/stencil_desc_ast_c++-naive.cpp"

using namespace dawn;

namespace sdesctest {

void test_01_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  if(cxxnaive::globals::get().var_runtime == 1) {
    for(int i = dom.iminus(); i < (dom.isize() - dom.iplus()); ++i) {
      for(int j = dom.jminus(); j < (dom.jsize() - dom.jplus()); ++j) {
        for(int k = dom.kminus(); k < (dom.ksize() - dom.kplus()); ++k) {

          out(i, j, k) = in(i, j, k) + cxxnaive::globals::get().var_runtime;
        }
      }
    }
  }
}
void test_02_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  if(cxxnaive::globals::get().var_compiletime == 2) {
    for(int i = dom.iminus(); i < (dom.isize() - dom.iplus()); ++i) {
      for(int j = dom.jminus(); j < (dom.jsize() - dom.jplus()); ++j) {
        for(int k = dom.kminus(); k < (dom.ksize() - dom.kplus()); ++k) {
          out(i, j, k) = in(i, j, k) + cxxnaive::globals::get().var_compiletime;
        }
      }
    }
  }
}
void test_03_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  if(cxxnaive::globals::get().var_runtime == 1) {
    if(cxxnaive::globals::get().var_compiletime == 2) {
      for(int i = dom.iminus(); i < (dom.isize() - dom.iplus()); ++i) {
        for(int j = dom.jminus(); j < (dom.jsize() - dom.jplus()); ++j) {
          for(int k = dom.kminus(); k < (dom.ksize() - dom.kplus()); ++k) {
            out(i, j, k) = in(i, j, k) + cxxnaive::globals::get().var_runtime + cxxnaive::globals::get().var_compiletime;
          }
        }
      }
    }
  }
}
void test_04_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  if(cxxnaive::globals::get().var_compiletime == 2) {
    if(cxxnaive::globals::get().var_compiletime != 1) {
      for(int i = dom.iminus(); i < (dom.isize() - dom.iplus()); ++i) {
        for(int j = dom.jminus(); j < (dom.jsize() - dom.jplus()); ++j) {
          for(int k = dom.kminus(); k < (dom.ksize() - dom.kplus()); ++k) {
            out(i, j, k) = 0.0;
          }
        }
      }
      if(cxxnaive::globals::get().var_compiletime == 2) {
        for(int i = dom.iminus(); i < (dom.isize() - dom.iplus()); ++i) {
          for(int j = dom.jminus(); j < (dom.jsize() - dom.jplus()); ++j) {
            for(int k = dom.kminus(); k < (dom.ksize() - dom.kplus()); ++k) {
              out(i, j, k) += 2 + in(i, j, k);
            }
          }
        }
      }
    }
  }
}
void test_05_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  if(cxxnaive::globals::get().var_compiletime == 2) {
    double some_var = 5.0;
    if(cxxnaive::globals::get().var_runtime < some_var) {
      for(int i = dom.iminus(); i < (dom.isize() - dom.iplus()); ++i) {
        for(int j = dom.jminus(); j < (dom.jsize() - dom.jplus()); ++j) {
          for(int k = dom.kminus(); k < (dom.ksize() - dom.kplus()); ++k) {
            out(i, j, k) = 2 * in(i, j, k);
          }
        }
      }
    }
  }
}
void test_06_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  if(cxxnaive::globals::get().var_compiletime == 2) {
    double some_var = 5.0;
    if(cxxnaive::globals::get().var_compiletime < some_var) {
      for(int i = dom.iminus(); i < (dom.isize() - dom.iplus()); ++i) {
        for(int j = dom.jminus(); j < (dom.jsize() - dom.jplus()); ++j) {
          for(int k = dom.kminus(); k < (dom.ksize() - dom.kplus()); ++k) {
            out(i, j, k) = 2 * in(i, j, k);
          }
        }
      }
    }
  }
}
void test_07_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  if(cxxnaive::globals::get().var_compiletime == 2) {
    double some_var = 5.0;
    double some_other_var = cxxnaive::globals::get().var_compiletime;

    some_var += 1.0;

    if((cxxnaive::globals::get().var_compiletime + some_var + some_other_var) == 10) {
      for(int i = dom.iminus(); i < (dom.isize() - dom.iplus()); ++i) {
        for(int j = dom.jminus(); j < (dom.jsize() - dom.jplus()); ++j) {
          for(int k = dom.kminus(); k < (dom.ksize() - dom.kplus()); ++k) {
            out(i, j, k) = 2 * in(i, j, k);
          }
        }
      }
    }
  }
}
void test_08_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  if(cxxnaive::globals::get().var_compiletime == 2) {
    for(int i = dom.iminus(); i < (dom.isize() - dom.iplus()); ++i) {
      for(int j = dom.jminus(); j < (dom.jsize() - dom.jplus()); ++j) {
        for(int k = dom.kminus(); k < (dom.ksize() - dom.kplus()); ++k) {
          out(i, j, k) = 4 * in(i, j, k);
        }
      }
    }
  }
}
void test_09_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  if(cxxnaive::globals::get().var_compiletime == 2) {
    if(cxxnaive::globals::get().var_compiletime == 2) {
      for(int i = dom.iminus(); i < (dom.isize() - dom.iplus()); ++i) {
        for(int j = dom.jminus(); j < (dom.jsize() - dom.jplus()); ++j) {
          for(int k = dom.kminus(); k < (dom.ksize() - dom.kplus()); ++k) {
            out(i, j, k) = 2 * in(i, j, k);
          }
        }
      }
    }
  }
}

} // namespace sdesctest

TEST(stencil_desc_ast, test_01) {
  domain dom(Options::getInstance().m_size[0], Options::getInstance().m_size[1], Options::getInstance().m_size[2]);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);

  verifier verif(dom);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize());
  storage_t in(meta_data, "in"), out_gt(meta_data, "out-gt"), out_naive(meta_data, "out-naive"), out_ref(meta_data, "out-ref");

  verif.fillMath(8.0, 2.0, 1.5, 1.5, 2.0, 4.0, in);
  verif.fill(-1.0, out_gt, out_naive);
  verif.fill(-2.0, out_ref);

  gridtools::test_01_stencil test_01_gt(dom, in, out_gt);
  cxxnaive::test_01_stencil test_01_naive(dom, in, out_naive);
  sdesctest::test_01_stencil_reference(dom, in, out_ref);

  test_01_gt.run();
  test_01_naive.run();

  ASSERT_TRUE(verif.verify(out_gt, out_naive));

  ASSERT_TRUE(verif.verify(out_naive, out_ref));
}

TEST(stencil_desc_ast, test_02) {
  domain dom(Options::getInstance().m_size[0], Options::getInstance().m_size[1], Options::getInstance().m_size[2]);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);

  verifier verif(dom);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize());
  storage_t in(meta_data, "in"), out_gt(meta_data, "out-gt"), out_naive(meta_data, "out-naive"), out_ref(meta_data, "out-ref");

  verif.fillMath(8.0, 2.0, 1.5, 1.5, 2.0, 4.0, in);
  verif.fill(-1.0, out_gt, out_naive);
  verif.fill(-2.0, out_ref);

  gridtools::test_02_stencil test_02_gt(dom, in, out_gt);
  cxxnaive::test_02_stencil test_02_naive(dom, in, out_naive);
  sdesctest::test_02_stencil_reference(dom, in, out_ref);

  test_02_gt.run();
  test_02_naive.run();

  ASSERT_TRUE(verif.verify(out_gt, out_naive));

  ASSERT_TRUE(verif.verify(out_naive, out_ref));
}

TEST(stencil_desc_ast, test_03) {
  domain dom(Options::getInstance().m_size[0], Options::getInstance().m_size[1], Options::getInstance().m_size[2]);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);

  verifier verif(dom);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize());
  storage_t in(meta_data, "in"), out_gt(meta_data, "out-gt"), out_naive(meta_data, "out-naive"), out_ref(meta_data, "out-ref");

  verif.fillMath(8.0, 2.0, 1.5, 1.5, 2.0, 4.0, in);
  verif.fill(-1.0, out_gt, out_naive);
  verif.fill(-2.0, out_ref);

  gridtools::test_03_stencil test_03_gt(dom, in, out_gt);
  cxxnaive::test_03_stencil test_03_naive(dom, in, out_naive);
  sdesctest::test_03_stencil_reference(dom, in, out_ref);

  test_03_gt.run();
  test_03_naive.run();

  ASSERT_TRUE(verif.verify(out_gt, out_naive));

  ASSERT_TRUE(verif.verify(out_naive, out_ref));
}

TEST(stencil_desc_ast, test_04) {
  domain dom(Options::getInstance().m_size[0], Options::getInstance().m_size[1], Options::getInstance().m_size[2]);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);

  verifier verif(dom);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize());
  storage_t in(meta_data, "in"), out_gt(meta_data, "out-gt"), out_naive(meta_data, "out-naive"), out_ref(meta_data, "out-ref");

  verif.fillMath(8.0, 2.0, 1.5, 1.5, 2.0, 4.0, in);
  verif.fill(-1.0, out_gt, out_naive);
  verif.fill(-2.0, out_ref);

  gridtools::test_04_stencil test_04_gt(dom, in, out_gt);
  cxxnaive::test_04_stencil test_04_naive(dom, in, out_naive);
  sdesctest::test_04_stencil_reference(dom, in, out_ref);

  test_04_gt.run();
  test_04_naive.run();

  ASSERT_TRUE(verif.verify(out_gt, out_naive));

  ASSERT_TRUE(verif.verify(out_naive, out_ref));
}

TEST(stencil_desc_ast, test_05) {
  domain dom(Options::getInstance().m_size[0], Options::getInstance().m_size[1], Options::getInstance().m_size[2]);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);

  verifier verif(dom);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize());
  storage_t in(meta_data, "in"), out_gt(meta_data, "out-gt"), out_naive(meta_data, "out-naive"), out_ref(meta_data, "out-ref");

  verif.fillMath(8.0, 2.0, 1.5, 1.5, 2.0, 4.0, in);
  verif.fill(-1.0, out_gt, out_naive);
  verif.fill(-2.0, out_ref);

  gridtools::test_05_stencil test_05_gt(dom, in, out_gt);
  cxxnaive::test_05_stencil test_05_naive(dom, in, out_naive);
  sdesctest::test_05_stencil_reference(dom, in, out_ref);

  test_05_gt.run();
  test_05_naive.run();

  ASSERT_TRUE(verif.verify(out_gt, out_naive));

  ASSERT_TRUE(verif.verify(out_naive, out_ref));
}
TEST(stencil_desc_ast, test_06) {
  domain dom(Options::getInstance().m_size[0], Options::getInstance().m_size[1], Options::getInstance().m_size[2]);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);

  verifier verif(dom);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize());
  storage_t in(meta_data, "in"), out_gt(meta_data, "out-gt"), out_naive(meta_data, "out-naive"), out_ref(meta_data, "out-ref");

  verif.fillMath(8.0, 2.0, 1.5, 1.5, 2.0, 4.0, in);
  verif.fill(-1.0, out_gt, out_naive);
  verif.fill(-2.0, out_ref);

  gridtools::test_06_stencil test_06_gt(dom, in, out_gt);
  cxxnaive::test_06_stencil test_06_naive(dom, in, out_naive);
  sdesctest::test_06_stencil_reference(dom, in, out_ref);

  test_06_gt.run();
  test_06_naive.run();

  ASSERT_TRUE(verif.verify(out_gt, out_naive));
  ASSERT_TRUE(verif.verify(out_naive, out_ref));
}
TEST(stencil_desc_ast, test_07) {
  domain dom(Options::getInstance().m_size[0], Options::getInstance().m_size[1], Options::getInstance().m_size[2]);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);

  verifier verif(dom);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize());
  storage_t in(meta_data, "in"), out_gt(meta_data, "out-gt"), out_naive(meta_data, "out-naive"), out_ref(meta_data, "out-ref");

  verif.fillMath(8.0, 2.0, 1.5, 1.5, 2.0, 4.0, in);
  verif.fill(-1.0, out_gt, out_naive);
  verif.fill(-2.0, out_ref);

  gridtools::test_07_stencil test_07_gt(dom, in, out_gt);
  cxxnaive::test_07_stencil test_07_naive(dom, in, out_naive);
  sdesctest::test_07_stencil_reference(dom, in, out_ref);

  test_07_gt.run();
  test_07_naive.run();

  ASSERT_TRUE(verif.verify(out_gt, out_naive));
  ASSERT_TRUE(verif.verify(out_naive, out_ref));
}

TEST(stencil_desc_ast, test_08) {
  domain dom(Options::getInstance().m_size[0], Options::getInstance().m_size[1], Options::getInstance().m_size[2]);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);

  verifier verif(dom);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize());
  storage_t in(meta_data, "in"), out_gt(meta_data, "out-gt"), out_naive(meta_data, "out-naive"), out_ref(meta_data, "out-ref");

  verif.fillMath(8.0, 2.0, 1.5, 1.5, 2.0, 4.0, in);
  verif.fill(-1.0, out_gt, out_naive);
  verif.fill(-2.0, out_ref);

  gridtools::test_08_stencil test_08_gt(dom, in, out_gt);
  cxxnaive::test_08_stencil test_08_naive(dom, in, out_naive);
  sdesctest::test_08_stencil_reference(dom, in, out_ref);

  test_08_gt.run();
  test_08_naive.run();

  ASSERT_TRUE(verif.verify(out_gt, out_naive));
  ASSERT_TRUE(verif.verify(out_naive, out_ref));
}

TEST(stencil_desc_ast, test_09) {
  domain dom(Options::getInstance().m_size[0], Options::getInstance().m_size[1], Options::getInstance().m_size[2]);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);

  verifier verif(dom);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize());
  storage_t in(meta_data, "in"), out_gt(meta_data, "out-gt"), out_naive(meta_data, "out-naive"), out_ref(meta_data, "out-ref");

  verif.fillMath(8.0, 2.0, 1.5, 1.5, 2.0, 4.0, in);
  verif.fill(-1.0, out_gt, out_naive);
  verif.fill(-2.0, out_ref);

  gridtools::test_09_stencil test_09_gt(dom, in, out_gt);
  cxxnaive::test_09_stencil test_09_naive(dom, in, out_naive);
  sdesctest::test_09_stencil_reference(dom, in, out_ref);

  test_09_gt.run();
  test_09_naive.run();

  ASSERT_TRUE(verif.verify(out_gt, out_naive));
  ASSERT_TRUE(verif.verify(out_naive, out_ref));
}
