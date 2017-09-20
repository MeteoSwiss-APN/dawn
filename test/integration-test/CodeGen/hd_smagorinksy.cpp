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
#include "gridtools/clang/math.hpp"
#include "gridtools/clang/verify.hpp"

using namespace gridtools::clang;

stencil_function avg {
  offset off;
  storage in;

  Do { return 0.5 * (in(off) + in); }
};

stencil_function delta {
  offset off;
  storage in;

  Do { return (in(off)-in); }
};

stencil_function laplacian {
  storage in, crlato, crlatu;

  Do {
    return in(i + 1) + in(i - 1) - 2.0 * in + crlato * delta(j + 1, in) + crlatu * delta(j - 1, in);
  }
};

stencil hd_smagorinsky_stencil {
  // Output fields
  storage u_out;
  storage v_out;

  // Input fields
  storage u_in;
  storage v_in;
  storage hdmaskvel;
  storage crlavo;
  storage crlavu;
  storage crlato;
  storage crlatu;
  storage acrlat0;

  // Scalar fields
  storage eddlon;
  storage eddlat;
  storage tau_smag;
  storage weight_smag;

  // Temporaries
  temporary_storage T_sqr_s, S_sqr_uv;

  Do {
    vertical_region(k_start, k_end) {
      const double frac_1_dx = acrlat0 * eddlon;
      const double frac_1_dy = eddlat / (double)6371.229e3;

      // Tension
      const double T_s = delta(j - 1, v_in) * frac_1_dy - delta(i - 1, u_in) * frac_1_dx;
      T_sqr_s = T_s * T_s;

      const double S_uv = delta(j + 1, u_in) * frac_1_dy + delta(i + 1, v_in) * frac_1_dx;
      S_sqr_uv = S_uv * S_uv;

      const double hdweight = weight_smag * hdmaskvel;

      // I direction
      double smag_u =
          tau_smag * math::sqrt((avg(i + 1, T_sqr_s) + avg(j - 1, S_sqr_uv))) - hdweight;
      smag_u = math::min(0.5, math::max(0.0, smag_u));

      // J direction
      double smag_v =
          tau_smag * math::sqrt((avg(j + 1, T_sqr_s) + avg(i - 1, S_sqr_uv))) - hdweight;
      smag_v = math::min(0.5, math::max(0.0, smag_v));

      const double lapu = laplacian(u_in, crlato, crlatu);
      const double lapv = laplacian(v_in, crlavo, crlavu);

      u_out = u_in + smag_u * lapu;
      v_out = v_in + smag_v * lapv;
    }
  }
};

int main() {
  domain dom(64, 64, 80);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize());
  meta_data_j_t meta_data_j(1, dom.jsize(), 1);
  meta_data_scalar_t meta_data_scalar(1, 1, 1);

  // Output fields
  storage_t u_out(meta_data, "v_out");
  storage_t v_out(meta_data, "u_out");

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
  storage_scalar_t eddlon(meta_data_scalar, "eddlon");
  storage_scalar_t eddlat(meta_data_scalar, "eddlat");
  storage_scalar_t tau_smag(meta_data_scalar, "tau_smag");
  storage_scalar_t weight_smag(meta_data_scalar, "weight_smag");

  // Fill the fields
  verifier verif(dom);
  verif.fill_random(u_out, v_out, u_in, v_in, hdmaskvel, crlavo, crlavu, crlato, crlatu, acrlat0,
                    eddlon, eddlat, tau_smag, weight_smag);

  // Assemble the stencil ...
  hd_smagorinsky_stencil hd_smagorinsky(dom, u_out, v_out, u_in, v_in, hdmaskvel, crlavo, crlavu,
                                        crlato, crlatu, acrlat0, eddlon, eddlat, tau_smag,
                                        weight_smag);

  // ... and run it
  hd_smagorinsky.run();
}
