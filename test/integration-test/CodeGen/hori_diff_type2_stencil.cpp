#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

stencil_function delta {
  offset off;
  storage data;

  double Do() { return data(off) - data; }
};

stencil_function laplacian {
  storage data, crlato, crlatu;

  Do {
    return data(i + 1) + data(i - 1) - 2.0 * data + crlato * delta(j + 1, data) +
           crlatu * delta(j - 1, data);
  }
};

stencil_function diffusive_flux_x {
  storage lap, data;

  Do {
    const double flx = delta(i + 1, lap);
    return (flx * delta(i + 1, data)) > 0.0 ? 0.0 : flx;
  }
};

stencil_function diffusive_flux_y {
  storage lap, data, crlato;

  Do {
    const double fly = crlato * delta(j + 1, lap);
    return (fly * delta(j + 1, data)) > 0.0 ? 0.0 : fly;
  }
};

stencil hori_diff_type2_stencil {
  storage out, in;
  storage_j crlato, crlatu;
  storage hdmask;
  var lap;

  Do {
    vertical_region(k_start, k_end) {
      lap = laplacian(in, crlato, crlatu);
      const double delta_flux_x =
          diffusive_flux_x(lap, in) - diffusive_flux_x(lap(i - 1), in(i - 1));
      const double delta_flux_y = diffusive_flux_y(lap, in, crlato) -
                                  diffusive_flux_y(lap(j - 1), in(j - 1), crlato(j - 1));
      out = in - hdmask * (delta_flux_x + delta_flux_y);
    }
  }
};
