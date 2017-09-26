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

// RUN: %gtclang% %file% -o%filename%_gen.cpp --config=%filedir%/stencil_desc_ast.json | %c++% %filename%_gen.cpp %gridtools_flags% -o%tmpdir%/%filename% | %tmpdir%/%filename%

#include "gridtools/clang_dsl.hpp"
#include "gridtools/clang/verify.hpp"
#include <iostream>

using namespace gridtools::clang;

globals {
  int var_runtime = 1;  // == 1
  int var_compiletime;  // == 2
};

#define IJK_LOOP()                                                                                 \
  for(int i = dom.iminus(); i < (dom.isize() - dom.iplus()); ++i)                                  \
    for(int j = dom.jminus(); j < (dom.jsize() - dom.jplus()); ++j)                                \
      for(int k = dom.kminus(); k < (dom.ksize() - dom.kplus()); ++k)

//
// Test 1
//
stencil test_01_stencil {
  storage in, out;

  Do {
    if(var_runtime == 1)
      vertical_region(k_start, k_end)
        out = in + var_runtime;
  }
};

void test_01_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  if(globals::get().var_runtime == 1) {
    IJK_LOOP() {
      out(i, j, k) = in(i, j, k) + globals::get().var_runtime;
    }
  }
}

//
// Test 2
//
stencil test_02_stencil {
  storage in, out;

  Do {
    if(var_compiletime == 2)
      vertical_region(k_start, k_end)
        out = in + var_compiletime;
  }
};

void test_02_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  if(globals::get().var_compiletime == 2) {
    IJK_LOOP() {
      out(i, j, k) = in(i, j, k) + globals::get().var_compiletime;
    }
  }
}

//
// Test 3
//
stencil test_03_stencil {
  storage in, out;

  Do {
    if(var_runtime == 1)
      if(var_compiletime == 2)
        vertical_region(k_start, k_end)
          out = in + var_runtime + var_compiletime;
  }
};

void test_03_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  if(globals::get().var_runtime == 1) {
    if(globals::get().var_compiletime == 2) {
      IJK_LOOP() {
        out(i, j, k) = in(i, j, k) + globals::get().var_runtime + globals::get().var_compiletime;
      }
    }
  }
}

//
// Test 4
//
stencil test_04_stencil {
  storage in, out;

  Do {
    if(var_compiletime == 2)
      if(var_compiletime != 1) {
        vertical_region(k_start, k_end)
          out = 0.0;
        if(var_compiletime != 1) {
          vertical_region(k_start, k_end)
            out += 2 + in;
        }
      }
  }
};

void test_04_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  if(globals::get().var_compiletime == 2) {
    if(globals::get().var_compiletime != 1) {
      IJK_LOOP() {
        out(i, j, k) = 0.0;
      }
      if(globals::get().var_compiletime == 2) {
        IJK_LOOP() {
          out(i, j, k) += 2 + in(i, j, k);
        }
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
    if(var_compiletime == 2) {
      double some_var = 5.0;
      if(var_runtime < some_var)
        vertical_region(k_start, k_end)
          out = 2 * in;
    }
  }
};

void test_05_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  if(globals::get().var_compiletime == 2) {
    double some_var = 5.0;
    if(globals::get().var_runtime < some_var) {
      IJK_LOOP() {
        out(i, j, k) = 2 * in(i, j, k);
      }
    }
  }
}

//
// Test 6
//
stencil test_06_stencil {
  storage in, out;

  Do {
    if(var_compiletime == 2) {
      double some_var = 5.0;
      if(var_compiletime < some_var)
        vertical_region(k_start, k_end)
          out = 2 * in;
    }
  }
};

void test_06_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  if(globals::get().var_compiletime == 2) {
    double some_var = 5.0;
    if(globals::get().var_compiletime < some_var) {
      IJK_LOOP() {
        out(i, j, k) = 2 * in(i, j, k);
      }
    }
  }
}

//
// Test 7
//
stencil test_07_stencil {
  storage in, out;

  Do {
    if(var_compiletime == 2) {
      double some_var = 5.0;
      double some_other_var = var_compiletime;
  
      some_var += 1.0;
      
      if((var_compiletime + some_var + some_other_var) == 10)
        vertical_region(k_start, k_end)
          out = 2 * in;
    }
  }
};

void test_07_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  if(globals::get().var_compiletime == 2) {
    double some_var = 5.0;
    double some_other_var = globals::get().var_compiletime;

    some_var += 1.0;

    if((globals::get().var_compiletime + some_var + some_other_var) == 10) {
      IJK_LOOP() {
        out(i, j, k) = 2 * in(i, j, k);
      }
    }
  }
}

//
// Test 8
//
stencil test_08_stencil {
  storage in, out;
  temporary_storage tmp;

  Do {
    if(var_compiletime == 2) {
      vertical_region(k_start, k_end)
        tmp = 2 * in;
    }
    if(var_compiletime == 2) {
      vertical_region(k_start, k_end)
        out = 2 * tmp;
    }
  }
};

void test_08_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  if(globals::get().var_compiletime == 2) {
    IJK_LOOP() {
      out(i, j, k) = 4 * in(i, j, k);
    }
  }
}

//
// Test 9
//
stencil test_09_stencil_call {
  storage in, out;

  Do {
    if(var_compiletime == 2) {
      vertical_region(k_start, k_end)
        out = 2 * in;
    }
  }
};

stencil test_09_stencil {
  storage in, out;

  Do {
    if(var_compiletime == 2) {
      test_09_stencil_call(in, out);
    }
  }
};

void test_09_stencil_reference(const domain& dom, storage_t& in_s, storage_t& out_s) {
  auto in = make_host_view(in_s);
  auto out = make_host_view(out_s);
  if(globals::get().var_compiletime == 2) {
    if(globals::get().var_compiletime == 2) {
      IJK_LOOP() {
        out(i, j, k) = 2 * in(i, j, k);
      }
    }
  }
}

#define TEST(test)                                                                                 \
  storage_t test##_in(meta_data, #test "_in");                                                     \
  storage_t test##_out(meta_data, #test "_out");                                                   \
  storage_t test##_out_ref(meta_data, #test "_out_ref");                                           \
  verif.for_each([&](int i, int j, int k) { return i + j + k; }, test##_in);                       \
  verif.fill(-1.0, test##_out, test##_out_ref);                                                    \
  test##_stencil_reference(dom, test##_in, test##_out_ref);                                        \
  test##_stencil test(dom, test##_in, test##_out);                                                 \
  test.run();                                                                                      \
  if(!verif.verify(test##_out, test##_out_ref)) {                                                  \
    std::cerr << " >>> " << #test << " FAILED!" << std::endl;                                      \
    return 1;                                                                                      \
  }

int main() {
  domain dom(64, 64, 80);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);
  verifier verif(dom);


  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize());
  globals::get().var_runtime = 1;

  TEST(test_01)
  TEST(test_02)
  TEST(test_03)
  TEST(test_04)
  TEST(test_05)
  TEST(test_06)
  TEST(test_07)
  TEST(test_08)
  TEST(test_09)

  return 0;
}
