#include "gtclang_dsl_defs/verify.hpp"
#include "gtclang_dsl_defs/gtclang_dsl.hpp"

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

stencil horizontal_diffusion_type2_stencil {
  storage out, in, crlato, crlatu, hdmask;
  var lap;

  Do {
    vertical_region(k_start, k_end) {
      lap = laplacian(in, crlato, crlatu);
      const double delta_flux_x = diffusive_flux_x(lap, in) -
                                  diffusive_flux_x(lap(i - 1), in(i - 1));
      const double delta_flux_y = diffusive_flux_y(lap, in, crlato) -
                                  diffusive_flux_y(lap(j - 1), in(j - 1), crlato(j - 1));
      out = in - hdmask * (delta_flux_x + delta_flux_y);
    }
  }
};

void horizontal_diffusion_type2_stencil_reference(const domain& dom, storage_t& out_s, storage_t& in_s,
                                                  storage_j_t& crlato_s, storage_j_t& crlatu_s,
                                                  storage_t& hdmask_s, storage_t& lap_s) {
  auto out = make_host_view(out_s);
  auto in = make_host_view(in_s);
  auto crlato = make_host_view(crlato_s);
  auto crlatu = make_host_view(crlatu_s);
  auto hdmask = make_host_view(hdmask_s);
  auto lap = make_host_view(crlatu_s);

  for(int k = dom.kminus(); k < (dom.ksize() - dom.kplus()); ++k) {
    for(int i = dom.iminus() - 1; i < (dom.isize() - dom.iplus() + 1); ++i) {
      for(int j = dom.jminus() - 1; j < (dom.jsize() - dom.jplus() + 1); ++j) {
        lap(i, j, k) = in(i + 1, j, k) + in(i - 1, j, k) - 2.0 * in(i, j, k) +
                       crlato(0, j, 0) * (in(i, j + 1, k) - in(i, j, k)) +
                       crlatu(0, j, 0) * (in(i, j - 1, k) - in(i, j, k));
      }
    }
    for(int i = dom.iminus(); i < (dom.isize() - dom.iplus()); ++i) {
      for(int j = dom.jminus(); j < (dom.jsize() - dom.jplus()); ++j) {

        double flux_x_lap_center_delta = lap(i + 1, j, k) - lap(i, j, k);
        double flux_x_lap_iminus_delta = lap(i, j, k) - lap(i - 1, j, k);
        double flux_x_center = flux_x_lap_center_delta * (in(i + 1, j, k) - in(i, j, k)) > 0.0
                                   ? 0.0
                                   : flux_x_lap_center_delta;
        double flux_x_iminus = flux_x_lap_iminus_delta * (in(i, j, k) - in(i - 1, j, k)) > 0.0
                                   ? 0.0
                                   : flux_x_lap_iminus_delta;

        double flux_y_lap_center_delta = crlato(i, j, k) * (lap(i, j + 1, k) - lap(i, j, k));
        double flux_y_lap_jminus_delta = crlato(i, j - 1, k) * (lap(i, j, k) - lap(i, j - 1, k));
        double flux_y_center = flux_y_lap_center_delta * (in(i, j + 1, k) - in(i, j, k)) > 0.0
                                   ? 0.0
                                   : flux_y_lap_center_delta;
        double flux_y_jminus = flux_y_lap_jminus_delta * (in(i, j, k) - in(i, j - 1, k)) > 0.0
                                   ? 0.0
                                   : flux_y_lap_jminus_delta;

        double delta_flux_x = flux_x_center - flux_x_iminus;
        double delta_flux_y = flux_y_center - flux_y_jminus;

        out(i, j, k) = in(i, j, k) - hdmask(i, j, k) * (delta_flux_x + delta_flux_y);
      }
    }
  }
}

int main() {
  domain dom(64, 64, 80);
  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);

  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize());
  meta_data_j_t meta_data_j(1, dom.jsize(), 1);

  // Output fields
  storage_t u_out(meta_data, "u_out");
  storage_t u_out_ref(meta_data, "u_out_ref");

  // Input fields
  storage_t u_in(meta_data, "u_in");
  storage_t lap(meta_data, "lap");
  storage_t hdmask(meta_data, "hdmask");
  storage_j_t crlato(meta_data_j, "crlato");
  storage_j_t crlatu(meta_data_j, "crlatu");

  verifier verif(dom);
  verif.fill_random(u_out, u_out_ref, u_in, lap, crlato, crlatu, hdmask);

  horizontal_diffusion_type2_stencil_reference(dom, u_out_ref, u_in, crlato, crlatu, hdmask, lap);

  horizontal_diffusion_type2_stencil hd(dom, u_out, u_in, crlato, crlatu, hdmask);
  hd.run();

  return !verif.verify(u_out, u_out_ref);
}
