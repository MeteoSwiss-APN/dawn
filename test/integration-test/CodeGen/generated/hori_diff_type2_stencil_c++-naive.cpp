// gtclang (0.0.1-0085b07-x86_64-linux-gnu-5.4.0)
// based on LLVM/Clang (3.8.0), Dawn (0.0.1)
// Generated on 2017-12-29  16:13:15

#define GRIDTOOLS_CLANG_GENERATED 1
#ifndef BOOST_RESULT_OF_USE_TR1
 #define BOOST_RESULT_OF_USE_TR1 1
#endif
#ifndef BOOST_NO_CXX11_DECLTYPE
 #define BOOST_NO_CXX11_DECLTYPE 1
#endif
#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

;

;

;

;

namespace cxxnaive {

class hori_diff_type2_stencil {
 private:
  template <class DataView>
  struct ParamWrapper {
    DataView dview_;
    std::array<int, DataView::storage_info_t::ndims> offsets_;

    ParamWrapper(DataView dview, std::array<int, DataView::storage_info_t::ndims> offsets)
        : dview_(dview), offsets_(offsets) {}
  };

  template <class StorageType0, class StorageType1>
  static double diffusive_flux_x_interval_start_0_end_0(const int i, const int j, const int k,
                                                        ParamWrapper<gridtools::data_view<StorageType0>> pw_lap,
                                                        ParamWrapper<gridtools::data_view<StorageType1>> pw_data) {
    gridtools::data_view<StorageType0> lap = pw_lap.dview_;
    auto lap_offsets = pw_lap.offsets_;
    gridtools::data_view<StorageType1> data = pw_data.dview_;
    auto data_offsets = pw_data.offsets_;
    const gridtools::clang::float_type __local_flx_10 = delta_i_plus_1_interval_start_0_end_0(
        i, j, k, ParamWrapper<gridtools::data_view<StorageType0>>(lap, std::array<int, 3>{0, 0, 0}));
    return (((__local_flx_10 * delta_i_plus_1_interval_start_0_end_0(
                                   i, j, k, ParamWrapper<gridtools::data_view<StorageType1>>(
                                                data, std::array<int, 3>{0, 0, 0}))) > (gridtools::clang::float_type)0)
                ? (gridtools::clang::float_type)0
                : __local_flx_10);
  }

  template <class StorageType0>
  static double delta_i_plus_1_interval_start_0_end_0(const int i, const int j, const int k,
                                                      ParamWrapper<gridtools::data_view<StorageType0>> pw_data) {
    gridtools::data_view<StorageType0> data = pw_data.dview_;
    auto data_offsets = pw_data.offsets_;
    return (data(i + 1 + data_offsets[0], j + 0 + data_offsets[1], k + 0 + data_offsets[2]) -
            data(i + 0 + data_offsets[0], j + 0 + data_offsets[1], k + 0 + data_offsets[2]));
  }

  template <class StorageType0, class StorageType1, class StorageType2>
  static double diffusive_flux_y_interval_start_0_end_0(const int i, const int j, const int k,
                                                        ParamWrapper<gridtools::data_view<StorageType0>> pw_lap,
                                                        ParamWrapper<gridtools::data_view<StorageType1>> pw_data,
                                                        ParamWrapper<gridtools::data_view<StorageType2>> pw_crlato) {
    gridtools::data_view<StorageType0> lap = pw_lap.dview_;
    auto lap_offsets = pw_lap.offsets_;
    gridtools::data_view<StorageType1> data = pw_data.dview_;
    auto data_offsets = pw_data.offsets_;
    gridtools::data_view<StorageType2> crlato = pw_crlato.dview_;
    auto crlato_offsets = pw_crlato.offsets_;
    const gridtools::clang::float_type __local_fly_17 =
        (crlato(i + 0 + crlato_offsets[0], j + 0 + crlato_offsets[1], k + 0 + crlato_offsets[2]) *
         delta_j_plus_1_interval_start_0_end_0(
             i, j, k, ParamWrapper<gridtools::data_view<StorageType0>>(lap, std::array<int, 3>{0, 0, 0})));
    return (((__local_fly_17 * delta_j_plus_1_interval_start_0_end_0(
                                   i, j, k, ParamWrapper<gridtools::data_view<StorageType1>>(
                                                data, std::array<int, 3>{0, 0, 0}))) > (gridtools::clang::float_type)0)
                ? (gridtools::clang::float_type)0
                : __local_fly_17);
  }

  template <class StorageType0>
  static double delta_j_plus_1_interval_start_0_end_0(const int i, const int j, const int k,
                                                      ParamWrapper<gridtools::data_view<StorageType0>> pw_data) {
    gridtools::data_view<StorageType0> data = pw_data.dview_;
    auto data_offsets = pw_data.offsets_;
    return (data(i + 0 + data_offsets[0], j + 1 + data_offsets[1], k + 0 + data_offsets[2]) -
            data(i + 0 + data_offsets[0], j + 0 + data_offsets[1], k + 0 + data_offsets[2]));
  }

  struct sbase {
    virtual void run() {}

    virtual ~sbase() {}
  };
  template <class StorageType0, class StorageType1, class StorageType2, class StorageType3>
  struct stencil_0 : public sbase {
    // //Members
    const gridtools::clang::domain& m_dom;
    StorageType0& m_out;
    StorageType1& m_in;
    StorageType2& m_crlato;
    StorageType3& m_hdmask;
    gridtools::clang::meta_data_t m_meta_data;
    gridtools::clang::storage_t m_lap;

   public:
    stencil_0(const gridtools::clang::domain& dom_, StorageType0& out_, StorageType1& in_, StorageType2& crlato_,
              StorageType3& hdmask_)
        : m_dom(dom_),
          m_out(out_),
          m_in(in_),
          m_crlato(crlato_),
          m_hdmask(hdmask_),
          m_meta_data(dom_.isize(), dom_.jsize(), dom_.ksize()),
          m_lap(m_meta_data) {}

    ~stencil_0() {}

    virtual void run() {
      gridtools::data_view<StorageType0> out = gridtools::make_host_view(m_out);
      gridtools::data_view<StorageType1> in = gridtools::make_host_view(m_in);
      gridtools::data_view<StorageType2> crlato = gridtools::make_host_view(m_crlato);
      gridtools::data_view<StorageType3> hdmask = gridtools::make_host_view(m_hdmask);
      gridtools::data_view<storage_t> lap = gridtools::make_host_view(m_lap);
      for (int k = 0; k <= (m_dom.ksize() == 0 ? 0 : (m_dom.ksize() - m_dom.kplus() - 1)); ++k) {
        for (int i = m_dom.iminus(); i <= m_dom.isize() - m_dom.iplus() - 1; ++i) {
          for (int j = m_dom.jminus(); j <= m_dom.jsize() - m_dom.jplus() - 1; ++j) {
            lap(i + 0, j + 0, k + 0) = in(i + 0, j + 0, k + 0);
          }
        }
        for (int i = m_dom.iminus(); i <= m_dom.isize() - m_dom.iplus() - 1; ++i) {
          for (int j = m_dom.jminus(); j <= m_dom.jsize() - m_dom.jplus() - 1; ++j) {
            const gridtools::clang::float_type __local_delta_flux_x_9 =
                (diffusive_flux_x_interval_start_0_end_0(
                     i, j, k,
                     ParamWrapper<gridtools::data_view<gridtools::clang::storage_t>>(lap, std::array<int, 3>{0, 0, 0}),
                     ParamWrapper<gridtools::data_view<StorageType1>>(in, std::array<int, 3>{0, 0, 0})) -
                 diffusive_flux_x_interval_start_0_end_0(
                     i, j, k,
                     ParamWrapper<gridtools::data_view<gridtools::clang::storage_t>>(lap, std::array<int, 3>{-1, 0, 0}),
                     ParamWrapper<gridtools::data_view<StorageType1>>(in, std::array<int, 3>{-1, 0, 0})));
            const gridtools::clang::float_type __local_delta_flux_y_16 =
                (diffusive_flux_y_interval_start_0_end_0(
                     i, j, k,
                     ParamWrapper<gridtools::data_view<gridtools::clang::storage_t>>(lap, std::array<int, 3>{0, 0, 0}),
                     ParamWrapper<gridtools::data_view<StorageType1>>(in, std::array<int, 3>{0, 0, 0}),
                     ParamWrapper<gridtools::data_view<StorageType2>>(crlato, std::array<int, 3>{0, 0, 0})) -
                 diffusive_flux_y_interval_start_0_end_0(
                     i, j, k,
                     ParamWrapper<gridtools::data_view<gridtools::clang::storage_t>>(lap, std::array<int, 3>{0, -1, 0}),
                     ParamWrapper<gridtools::data_view<StorageType1>>(in, std::array<int, 3>{0, -1, 0}),
                     ParamWrapper<gridtools::data_view<StorageType2>>(crlato, std::array<int, 3>{0, -1, 0})));
            out(i + 0, j + 0, k + 0) =
                (in(i + 0, j + 0, k + 0) - (hdmask(i + 0, j + 0, k + 0) * __local_delta_flux_y_16));
          }
        }
      }
    }
  };
  static constexpr const char* s_name = "hori_diff_type2_stencil";
  sbase* m_stencil_0;

 public:
  hori_diff_type2_stencil(const hori_diff_type2_stencil&) = delete;

  // Members

  template <class StorageType1, class StorageType2, class StorageType3, class StorageType4, class StorageType5>
  hori_diff_type2_stencil(const gridtools::clang::domain& dom, StorageType1& out, StorageType2& in,
                          StorageType3& crlato, StorageType4& crlatu, StorageType5& hdmask)
      : m_stencil_0(
            new stencil_0<StorageType1, StorageType2, StorageType3, StorageType5>(dom, out, in, crlato, hdmask)) {}

  void run() { m_stencil_0->run(); }
};
}  // namespace cxxnaiv
;

// void horizontal_diffusion_type2_stencil_reference(const domain& dom, storage_t& out_s, storage_t& in_s,
//                                                  storage_j_t& crlato_s, storage_j_t& crlatu_s,
//                                                  storage_t& hdmask_s, storage_t& lap_s) {
//  auto out = make_host_view(out_s);
//  auto in = make_host_view(in_s);
//  auto crlato = make_host_view(crlato_s);
//  auto crlatu = make_host_view(crlatu_s);
//  auto hdmask = make_host_view(hdmask_s);
//  auto lap = make_host_view(crlatu_s);

//  for(int k = dom.kminus(); k < (dom.ksize() - dom.kplus()); ++k) {
//    for(int i = dom.iminus() - 1; i < (dom.isize() - dom.iplus() + 1); ++i) {
//      for(int j = dom.jminus() - 1; j < (dom.jsize() - dom.jplus() + 1); ++j) {
//        lap(i, j, k) = in(i + 1, j, k) + in(i - 1, j, k) - 2.0 * in(i, j, k) +
//                       crlato(0, j, 0) * (in(i, j + 1, k) - in(i, j, k)) +
//                       crlatu(0, j, 0) * (in(i, j - 1, k) - in(i, j, k));
//      }
//    }
//    for(int i = dom.iminus(); i < (dom.isize() - dom.iplus()); ++i) {
//      for(int j = dom.jminus(); j < (dom.jsize() - dom.jplus()); ++j) {

//        double flux_x_lap_center_delta = lap(i + 1, j, k) - lap(i, j, k);
//        double flux_x_lap_iminus_delta = lap(i, j, k) - lap(i - 1, j, k);
//        double flux_x_center = flux_x_lap_center_delta * (in(i + 1, j, k) - in(i, j, k)) > 0.0
//                                   ? 0.0
//                                   : flux_x_lap_center_delta;
//        double flux_x_iminus = flux_x_lap_iminus_delta * (in(i, j, k) - in(i - 1, j, k)) > 0.0
//                                   ? 0.0
//                                   : flux_x_lap_iminus_delta;

//        double flux_y_lap_center_delta = crlato(i, j, k) * (lap(i, j + 1, k) - lap(i, j, k));
//        double flux_y_lap_jminus_delta = crlato(i, j - 1, k) * (lap(i, j, k) - lap(i, j - 1, k));
//        double flux_y_center = flux_y_lap_center_delta * (in(i, j + 1, k) - in(i, j, k)) > 0.0
//                                   ? 0.0
//                                   : flux_y_lap_center_delta;
//        double flux_y_jminus = flux_y_lap_jminus_delta * (in(i, j, k) - in(i, j - 1, k)) > 0.0
//                                   ? 0.0
//                                   : flux_y_lap_jminus_delta;

//        double delta_flux_x = flux_x_center - flux_x_iminus;
//        double delta_flux_y = flux_y_center - flux_y_jminus;

//        out(i, j, k) = in(i, j, k) - hdmask(i, j, k) * (delta_flux_x + delta_flux_y);
//      }
//    }
//  }
//}

// int main() {
//  domain dom(64, 64, 80);
//  dom.set_halos(halo::value, halo::value, halo::value, halo::value, 0, 0);

//  meta_data_t meta_data(dom.isize(), dom.jsize(), dom.ksize());
//  meta_data_j_t meta_data_j(1, dom.jsize(), 1);

//  // Output fields
//  storage_t u_out(meta_data, "u_out");
//  storage_t u_out_ref(meta_data, "u_out_ref");

//  // Input fields
//  storage_t u_in(meta_data, "u_in");
//  storage_t lap(meta_data, "lap");
//  storage_t hdmask(meta_data, "hdmask");
//  storage_j_t crlato(meta_data_j, "crlato");
//  storage_j_t crlatu(meta_data_j, "crlatu");

//  verifier verif(dom);
//  verif.fill_random(u_out, u_out_ref, u_in, lap, crlato, crlatu, hdmask);

//  horizontal_diffusion_type2_stencil_reference(dom, u_out_ref, u_in, crlato, crlatu, hdmask, lap);

//  horizontal_diffusion_type2_stencil hd(dom, u_out, u_in, crlato, crlatu, hdmask);
//  hd.run();

//  return !verif.verify(u_out, u_out_ref);
//}
