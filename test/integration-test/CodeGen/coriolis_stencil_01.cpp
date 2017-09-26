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

// RUN: %gtclang% %file% -o%filename%_gen.cpp | %c++% %filename%_gen.cpp %gridtools_flags% -o%tmpdir%/%filename% | %tmpdir%/%filename%

#include "gridtools/clang_dsl.hpp"
#include "gridtools/clang/verify.hpp"

using namespace gridtools::clang;

stencil coriolis_stencil {
  storage u_tens, u_nnow, v_tens, v_nnow, fc;

  Do {
    vertical_region(k_start, k_end) {
      double z_fv_north = fc * (v_nnow + v_nnow(i + 1));
      double z_fv_south = fc(j - 1) * (v_nnow(j - 1) + v_nnow(i + 1, j - 1));
      u_tens += 0.25 * (z_fv_north + z_fv_south);

      double z_fu_east = fc * (u_nnow + u_nnow(j + 1));
      double z_fu_west = fc(i - 1) * (u_nnow(i - 1) + u_nnow(i - 1, j + 1));
      v_tens -= 0.25 * (z_fu_east + z_fu_west);
    }
  }
};

void coriolis_stencil_reference(const domain& dom, storage_t& u_tens_s, storage_t& u_nnow_s,
                                storage_t& v_tens_s, storage_t& v_nnow_s, storage_t& fc_s) {
  auto u_tens = make_host_view(u_tens_s);
  auto u_nnow = make_host_view(u_nnow_s);  
  auto v_tens = make_host_view(v_tens_s);
  auto v_nnow = make_host_view(v_nnow_s);
  auto fc = make_host_view(fc_s);

  for(int i = dom.iminus(); i < (dom.isize() - dom.iplus()); ++i) {
    for(int j = dom.jminus(); j < (dom.jsize() - dom.jplus()); ++j) {
      for(int k = dom.kminus(); k < (dom.ksize() - dom.kplus()); ++k) {
        double z_fv_north = fc(i, j, k) * (v_nnow(i, j, k) + v_nnow(i + 1, j, k));
        double z_fv_south = fc(i, j - 1, k) * (v_nnow(i, j - 1, k) + v_nnow(i + 1, j - 1, k));
        u_tens(i, j, k) += 0.25 * (z_fv_north + z_fv_south);

        double z_fu_east = fc(i, j, k) * (u_nnow(i, j, k) + u_nnow(i, j + 1, k));
        double z_fu_west = fc(i - 1, j, k) * (u_nnow(i - 1, j, k) + u_nnow(i - 1, j + 1, k));
        v_tens(i, j, k) -= 0.25 * (z_fu_east + z_fu_west);
      }
    }
  }
}

int main() {
  domain dom(64, 64, 80);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);
  verifier verif(dom);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize());
  storage_t u_tens(meta_data, "u_tens"), u_tens_ref(meta_data, "u_tens_ref");
  storage_t v_tens(meta_data, "v_tens"), v_tens_ref(meta_data, "v_tens_ref");
  storage_t u_nnow(meta_data, "u_nnow"), v_nnow(meta_data, "v_nnow"), fc(meta_data, "fc");

  verif.fill(-1.0, u_tens, u_tens_ref, v_tens, v_tens_ref, fc);
  verif.for_each([&](int i, int j, int k) { return i + j + k; }, u_nnow);
  verif.for_each([&](int i, int j, int k) { return -i - j - k; }, v_nnow);

  coriolis_stencil_reference(dom, u_tens_ref, u_nnow, v_tens_ref, v_nnow, fc);
 
  coriolis_stencil coriolis(dom, u_tens, u_nnow, v_tens, v_nnow, fc);
  coriolis.run();

  return !(verif.verify(u_tens, u_tens_ref) && verif.verify(v_tens, v_tens_ref));
}
