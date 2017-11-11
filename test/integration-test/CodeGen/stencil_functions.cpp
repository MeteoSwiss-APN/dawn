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

// RUN: %gtclang% %file% -o%filename%_gen.cpp -inline=pc | %c++% %filename%_gen.cpp %gridtools_flags% -o%tmpdir%/%filename% | %tmpdir%/%filename%

#include "gridtools/clang_dsl.hpp"
#include "gridtools/clang/verify.hpp"
#include <iostream>

using namespace gridtools::clang;

stencil_function delta {
  offset off;
  storage in;

  Do { return in(off)-in; }
};

//
// Test 1
//

stencil test_01_stencil {
  storage in, out;

  Do {
    vertical_region(k_start, k_end)
        out = delta(i + 1, in);
  }
};

void test_01_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  for(int i = dom.iminus(); i < (dom.isize() - dom.iplus()); ++i) {
    for(int j = dom.jminus(); j < (dom.jsize() - dom.jplus()); ++j) {
      for(int k = dom.kminus(); k < (dom.ksize() - dom.kplus()); ++k) {
        out(i, j, k) = in(i + 1, j, k) - in(i, j, k);
      }
    }
  }
}

//
// Test 2
//

stencil test_02_stencil {
  storage in, out;

  Do {
    vertical_region(k_start, k_end)
        out = delta(i + 1, in) + delta(j + 1, in);
  }
};

void test_02_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  for(int i = dom.iminus(); i < (dom.isize() - dom.iplus()); ++i) {
    for(int j = dom.jminus(); j < (dom.jsize() - dom.jplus()); ++j) {
      for(int k = dom.kminus(); k < (dom.ksize() - dom.kplus()); ++k) {
        out(i, j, k) = in(i + 1, j, k) - in(i, j, k) + in(i, j + 1, k) - in(i, j, k);
      }
    }
  }
}

//
// Test 3
//

stencil_function delta_nested {
  offset off;
  storage in;

  Do { return delta(off, in); }
};

stencil test_03_stencil {
  storage in, out;

  Do {
    vertical_region(k_start, k_end)
        out = delta_nested(i + 1, in);
  }
};

void test_03_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  for(int i = dom.iminus(); i < (dom.isize() - dom.iplus()); ++i) {
    for(int j = dom.jminus(); j < (dom.jsize() - dom.jplus()); ++j) {
      for(int k = dom.kminus(); k < (dom.ksize() - dom.kplus()); ++k) {
        out(i, j, k) = in(i + 1, j, k) - in(i, j, k);
      }
    }
  }
}

//
// Test 4
//

stencil_function sum {
  storage s1, s2;

  Do { return s1 + s2; }
};

stencil_function delta_sum {
  offset off1;
  offset off2;
  storage in;

  Do { return sum(delta(off1, in), delta(off2, in)); }
};

stencil test_04_stencil {
  storage in, out;

  Do {
    vertical_region(k_start, k_end)
        out = delta_sum(i + 1, j + 1, in);
  }
};

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

//
// Test 5
//

stencil test_05_stencil {
  storage in, out;

  Do {
    vertical_region(k_start, k_end)
        out = delta(i + 1, delta(j + 1, delta(i + 1, in)));
  }
};

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

//
// Test 6
//

stencil_function layer_1_ret {
  storage in;

  Do { return in; };
};

stencil_function layer_2_ret {
  storage in;

  Do {
    return layer_1_ret(in);
  }
};

stencil_function layer_3_ret {
  storage in;

  Do { return layer_2_ret(in); }
};

stencil test_06_stencil {
  storage in, out;

  Do {
    vertical_region(k_start, k_end)
        out = layer_3_ret(in);
  }
};

void test_06_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  for(int i = dom.iminus(); i < (dom.isize() - dom.iplus()); ++i) {
    for(int j = dom.jminus(); j < (dom.jsize() - dom.jplus()); ++j) {
      for(int k = dom.kminus(); k < (dom.ksize() - dom.kplus()); ++k) {
        out(i, j, k) = in(i, j, k);
      }
    }
  }
}

//
// Test 7
//

stencil_function layer_1_no_ret {
  storage out, in;

  Do { out = in; };
};

stencil_function layer_2_no_ret {
  storage out, in;

  Do { layer_1_no_ret(out, in); }
};

stencil_function layer_3_no_ret {
  storage out, in;

  Do { layer_2_no_ret(out, in); }
};

stencil test_07_stencil {
  storage in, out;

  Do {
    vertical_region(k_start, k_end)
        layer_3_no_ret(out, in);
  }
};

void test_07_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  for(int i = dom.iminus(); i < (dom.isize() - dom.iplus()); ++i) {
    for(int j = dom.jminus(); j < (dom.jsize() - dom.jplus()); ++j) {
      for(int k = dom.kminus(); k < (dom.ksize() - dom.kplus()); ++k) {
        out(i, j, k) = in(i, j, k);
      }
    }
  }
}

#define TEST(test)                                                           \
  storage_t test##_in(meta_data, #test "_in");                               \
  storage_t test##_out(meta_data, #test "_out");                             \
  storage_t test##_out_ref(meta_data, #test "_out_ref");                     \
  verif.for_each([&](int i, int j, int k) { return i + j + k; }, test##_in); \
  verif.fill(-1.0, test##_out, test##_out_ref);                              \
  test##_stencil_reference(dom, test##_in, test##_out_ref);                  \
  test##_stencil test(dom, test##_in, test##_out);                           \
  test.run();                                                                \
  if(!verif.verify(test##_out, test##_out_ref)) {                            \
    std::cerr << " >>> " << #test << " FAILED!" << std::endl;                \
    return 1;                                                                \
  }

int main() {
  domain dom(64, 64, 80);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);
  verifier verif(dom);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize());

  TEST(test_01)
  TEST(test_02)
  TEST(test_03)
  TEST(test_04)
  TEST(test_05)
  TEST(test_06)
  TEST(test_07)

  return 0;
}
