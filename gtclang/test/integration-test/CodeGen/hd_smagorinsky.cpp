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

#include "gtclang_dsl_defs/math.hpp"
#include "gtclang_dsl_defs/gtclang_dsl.hpp"

using namespace gtclang::dsl;

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
  storage_j crlavo;
  storage_j crlavu;
  storage_j crlato;
  storage_j crlatu;
  storage_j acrlat0;

  // Scalar fields
  storage eddlon;
  storage eddlat;
  storage tau_smag;
  storage weight_smag;

  // Temporaries
  var T_sqr_s, S_sqr_uv;

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
