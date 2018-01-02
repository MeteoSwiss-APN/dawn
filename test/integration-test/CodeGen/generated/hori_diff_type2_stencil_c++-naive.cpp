// gtclang (0.0.1-a7a9177-x86_64-linux-gnu-5.4.0)
// based on LLVM/Clang (3.8.0), Dawn (0.0.1)
// Generated on 2018-01-02  00:58:45

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
template <size_t N>
std::array<int, N> operator+(std::array<int, N> const& a, std::array<int, N> const& b) {
  std::array<int, N> res;
  for (size_t i = 0; i < N; ++i) {
    res[i] = a[i] + b[i];
  }
  return res;
}

class hori_diff_type2_stencil {
 private:
  template <class DataView>
  struct ParamWrapper {
    DataView dview_;
    std::array<int, DataView::storage_info_t::ndims> offsets_;

    ParamWrapper(DataView dview, std::array<int, DataView::storage_info_t::ndims> offsets)
        : dview_(dview), offsets_(offsets) {}
  };

  template <class StorageType0, class StorageType1, class StorageType2>
  static double laplacian_interval_start_0_end_0(const int i, const int j, const int k,
                                                 ParamWrapper<gridtools::data_view<StorageType0>> pw_data,
                                                 ParamWrapper<gridtools::data_view<StorageType1>> pw_crlato,
                                                 ParamWrapper<gridtools::data_view<StorageType2>> pw_crlatu) {
    gridtools::data_view<StorageType0> data = pw_data.dview_;
    auto data_offsets = pw_data.offsets_;
    gridtools::data_view<StorageType1> crlato = pw_crlato.dview_;
    auto crlato_offsets = pw_crlato.offsets_;
    gridtools::data_view<StorageType2> crlatu = pw_crlatu.dview_;
    auto crlatu_offsets = pw_crlatu.offsets_;
    return ((((data(i + 1 + data_offsets[0], j + 0 + data_offsets[1], k + 0 + data_offsets[2]) +
               data(i + -1 + data_offsets[0], j + 0 + data_offsets[1], k + 0 + data_offsets[2])) -
              ((gridtools::clang::float_type)2 *
               data(i + 0 + data_offsets[0], j + 0 + data_offsets[1], k + 0 + data_offsets[2]))) +
             (crlato(i + 0 + crlato_offsets[0], j + 0 + crlato_offsets[1], k + 0 + crlato_offsets[2]) *
              delta_j_plus_1_interval_start_0_end_0(i, j, k, ParamWrapper<gridtools::data_view<StorageType0>>(
                                                                 data, std::array<int, 3>{0, 0, 0} + data_offsets)))) +
            (crlatu(i + 0 + crlatu_offsets[0], j + 0 + crlatu_offsets[1], k + 0 + crlatu_offsets[2]) *
             delta_j_minus_1_interval_start_0_end_0(i, j, k, ParamWrapper<gridtools::data_view<StorageType0>>(
                                                                 data, std::array<int, 3>{0, 0, 0} + data_offsets))));
  }

  template <class StorageType0>
  static double delta_j_plus_1_interval_start_0_end_0(const int i, const int j, const int k,
                                                      ParamWrapper<gridtools::data_view<StorageType0>> pw_data) {
    gridtools::data_view<StorageType0> data = pw_data.dview_;
    auto data_offsets = pw_data.offsets_;
    return (data(i + 0 + data_offsets[0], j + 1 + data_offsets[1], k + 0 + data_offsets[2]) -
            data(i + 0 + data_offsets[0], j + 0 + data_offsets[1], k + 0 + data_offsets[2]));
  }

  template <class StorageType0>
  static double delta_j_minus_1_interval_start_0_end_0(const int i, const int j, const int k,
                                                       ParamWrapper<gridtools::data_view<StorageType0>> pw_data) {
    gridtools::data_view<StorageType0> data = pw_data.dview_;
    auto data_offsets = pw_data.offsets_;
    return (data(i + 0 + data_offsets[0], j + -1 + data_offsets[1], k + 0 + data_offsets[2]) -
            data(i + 0 + data_offsets[0], j + 0 + data_offsets[1], k + 0 + data_offsets[2]));
  }

  template <class StorageType0, class StorageType1>
  static double diffusive_flux_x_interval_start_0_end_0(const int i, const int j, const int k,
                                                        ParamWrapper<gridtools::data_view<StorageType0>> pw_lap,
                                                        ParamWrapper<gridtools::data_view<StorageType1>> pw_data) {
    gridtools::data_view<StorageType0> lap = pw_lap.dview_;
    auto lap_offsets = pw_lap.offsets_;
    gridtools::data_view<StorageType1> data = pw_data.dview_;
    auto data_offsets = pw_data.offsets_;
    const gridtools::clang::float_type __local_flx_11 = delta_i_plus_1_interval_start_0_end_0(
        i, j, k, ParamWrapper<gridtools::data_view<StorageType0>>(lap, std::array<int, 3>{0, 0, 0} + lap_offsets));
    return (((__local_flx_11 *
              delta_i_plus_1_interval_start_0_end_0(i, j, k, ParamWrapper<gridtools::data_view<StorageType1>>(
                                                                 data, std::array<int, 3>{0, 0, 0} + data_offsets))) >
             (gridtools::clang::float_type)0)
                ? (gridtools::clang::float_type)0
                : __local_flx_11);
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
    const gridtools::clang::float_type __local_fly_18 =
        (crlato(i + 0 + crlato_offsets[0], j + 0 + crlato_offsets[1], k + 0 + crlato_offsets[2]) *
         delta_j_plus_1_interval_start_0_end_0(i, j, k, ParamWrapper<gridtools::data_view<StorageType0>>(
                                                            lap, std::array<int, 3>{0, 0, 0} + lap_offsets)));
    return (((__local_fly_18 *
              delta_j_plus_1_interval_start_0_end_0(i, j, k, ParamWrapper<gridtools::data_view<StorageType1>>(
                                                                 data, std::array<int, 3>{0, 0, 0} + data_offsets))) >
             (gridtools::clang::float_type)0)
                ? (gridtools::clang::float_type)0
                : __local_fly_18);
  }

  struct sbase {
    virtual void run() {}

    virtual ~sbase() {}
  };
  template <class StorageType0, class StorageType1, class StorageType2, class StorageType3, class StorageType4>
  struct stencil_0 : public sbase {
    // //Members
    const gridtools::clang::domain& m_dom;
    StorageType0& m_out;
    StorageType1& m_in;
    StorageType2& m_crlato;
    StorageType3& m_crlatu;
    StorageType4& m_hdmask;
    gridtools::clang::meta_data_t m_meta_data;
    gridtools::clang::storage_t m_lap;

   public:
    stencil_0(const gridtools::clang::domain& dom_, StorageType0& out_, StorageType1& in_, StorageType2& crlato_,
              StorageType3& crlatu_, StorageType4& hdmask_)
        : m_dom(dom_),
          m_out(out_),
          m_in(in_),
          m_crlato(crlato_),
          m_crlatu(crlatu_),
          m_hdmask(hdmask_),
          m_meta_data(dom_.isize(), dom_.jsize(), dom_.ksize()),
          m_lap(m_meta_data) {}

    ~stencil_0() {}

    virtual void run() {
      gridtools::data_view<StorageType0> out = gridtools::make_host_view(m_out);
      std::array<int, 3> out_offsets{0, 0, 0};
      gridtools::data_view<StorageType1> in = gridtools::make_host_view(m_in);
      std::array<int, 3> in_offsets{0, 0, 0};
      gridtools::data_view<StorageType2> crlato = gridtools::make_host_view(m_crlato);
      std::array<int, 3> crlato_offsets{0, 0, 0};
      gridtools::data_view<StorageType3> crlatu = gridtools::make_host_view(m_crlatu);
      std::array<int, 3> crlatu_offsets{0, 0, 0};
      gridtools::data_view<StorageType4> hdmask = gridtools::make_host_view(m_hdmask);
      std::array<int, 3> hdmask_offsets{0, 0, 0};
      gridtools::data_view<storage_t> lap = gridtools::make_host_view(m_lap);
      std::array<int, 3> lap_offsets{0, 0, 0};
      for (int k = 0 + 0; k <= (m_dom.ksize() == 0 ? 0 : (m_dom.ksize() - m_dom.kplus() - 1)) + 0; ++k) {
        for (int i = m_dom.iminus() + -1; i <= m_dom.isize() - m_dom.iplus() - 1 + 1; ++i) {
          for (int j = m_dom.jminus() + -1; j <= m_dom.jsize() - m_dom.jplus() - 1 + 1; ++j) {
            lap(i + 0, j + 0, k + 0) = laplacian_interval_start_0_end_0(
                i, j, k, ParamWrapper<gridtools::data_view<StorageType1>>(in, std::array<int, 3>{0, 0, 0} + in_offsets),
                ParamWrapper<gridtools::data_view<StorageType2>>(crlato, std::array<int, 3>{0, 0, 0} + crlato_offsets),
                ParamWrapper<gridtools::data_view<StorageType3>>(crlatu, std::array<int, 3>{0, 0, 0} + crlatu_offsets));
          }
        }
        for (int i = m_dom.iminus() + 0; i <= m_dom.isize() - m_dom.iplus() - 1 + 0; ++i) {
          for (int j = m_dom.jminus() + 0; j <= m_dom.jsize() - m_dom.jplus() - 1 + 0; ++j) {
            const gridtools::clang::float_type __local_delta_flux_x_10 =
                (diffusive_flux_x_interval_start_0_end_0(
                     i, j, k, ParamWrapper<gridtools::data_view<gridtools::clang::storage_t>>(
                                  lap, std::array<int, 3>{0, 0, 0} + lap_offsets),
                     ParamWrapper<gridtools::data_view<StorageType1>>(in, std::array<int, 3>{0, 0, 0} + in_offsets)) -
                 diffusive_flux_x_interval_start_0_end_0(
                     i, j, k, ParamWrapper<gridtools::data_view<gridtools::clang::storage_t>>(
                                  lap, std::array<int, 3>{-1, 0, 0} + lap_offsets),
                     ParamWrapper<gridtools::data_view<StorageType1>>(in, std::array<int, 3>{-1, 0, 0} + in_offsets)));
            const gridtools::clang::float_type __local_delta_flux_y_17 =
                (diffusive_flux_y_interval_start_0_end_0(
                     i, j, k, ParamWrapper<gridtools::data_view<gridtools::clang::storage_t>>(
                                  lap, std::array<int, 3>{0, 0, 0} + lap_offsets),
                     ParamWrapper<gridtools::data_view<StorageType1>>(in, std::array<int, 3>{0, 0, 0} + in_offsets),
                     ParamWrapper<gridtools::data_view<StorageType2>>(crlato,
                                                                      std::array<int, 3>{0, 0, 0} + crlato_offsets)) -
                 diffusive_flux_y_interval_start_0_end_0(
                     i, j, k, ParamWrapper<gridtools::data_view<gridtools::clang::storage_t>>(
                                  lap, std::array<int, 3>{0, -1, 0} + lap_offsets),
                     ParamWrapper<gridtools::data_view<StorageType1>>(in, std::array<int, 3>{0, -1, 0} + in_offsets),
                     ParamWrapper<gridtools::data_view<StorageType2>>(crlato,
                                                                      std::array<int, 3>{0, -1, 0} + crlato_offsets)));
            out(i + 0, j + 0, k + 0) =
                (in(i + 0, j + 0, k + 0) -
                 (hdmask(i + 0, j + 0, k + 0) * (__local_delta_flux_x_10 + __local_delta_flux_y_17)));
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
      : m_stencil_0(new stencil_0<StorageType1, StorageType2, StorageType3, StorageType4, StorageType5>(
            dom, out, in, crlato, crlatu, hdmask)) {}

  void run() { m_stencil_0->run(); }
};
}  // namespace cxxnaiv
;
