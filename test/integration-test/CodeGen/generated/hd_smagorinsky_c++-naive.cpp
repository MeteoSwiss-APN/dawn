// gtclang (0.0.1-c392bf6-x86_64-linux-gnu-5.4.0)
// based on LLVM/Clang (3.8.0), Dawn (0.0.1)
// Generated on 2017-12-28  01:03:29

#define GRIDTOOLS_CLANG_GENERATED 1
#define GRIDTOOLS_CLANG_HALO_EXTEND 3
#ifndef BOOST_RESULT_OF_USE_TR1
 #define BOOST_RESULT_OF_USE_TR1 1
#endif
#ifndef BOOST_NO_CXX11_DECLTYPE
 #define BOOST_NO_CXX11_DECLTYPE 1
#endif
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

#include "gridtools/clang/math.hpp"
#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

;

;

;

namespace cxxnaive {

class hd_smagorinsky_stencil {
 private:
  template <class DataView>
  struct ParamWrapper {
    DataView dview_;
    std::array<int, DataView::storage_info_t::ndims> offsets_;

    ParamWrapper(DataView dview, std::array<int, DataView::storage_info_t::ndims> offsets)
        : dview_(dview), offsets_(offsets) {}
  };

  template <class StorageType0>
  static double delta_j_minus_1_interval_start_0_end_0(const int i, const int j, const int k,
                                                       ParamWrapper<gridtools::data_view<StorageType0>> pw_in) {
    gridtools::data_view<StorageType0> in = pw_in.dview_;
    auto in_offsets = pw_in.offsets_;
    return (in(i + 0 + in_offsets[0], j + -1 + in_offsets[1], k + 0 + in_offsets[2]) -
            in(i + 0 + in_offsets[0], j + 0 + in_offsets[1], k + 0 + in_offsets[2]));
  }

  template <class StorageType0>
  static double delta_i_minus_1_interval_start_0_end_0(const int i, const int j, const int k,
                                                       ParamWrapper<gridtools::data_view<StorageType0>> pw_in) {
    gridtools::data_view<StorageType0> in = pw_in.dview_;
    auto in_offsets = pw_in.offsets_;
    return (in(i + -1 + in_offsets[0], j + 0 + in_offsets[1], k + 0 + in_offsets[2]) -
            in(i + 0 + in_offsets[0], j + 0 + in_offsets[1], k + 0 + in_offsets[2]));
  }

  template <class StorageType0>
  static double delta_j_plus_1_interval_start_0_end_0(const int i, const int j, const int k,
                                                      ParamWrapper<gridtools::data_view<StorageType0>> pw_in) {
    gridtools::data_view<StorageType0> in = pw_in.dview_;
    auto in_offsets = pw_in.offsets_;
    return (in(i + 0 + in_offsets[0], j + 1 + in_offsets[1], k + 0 + in_offsets[2]) -
            in(i + 0 + in_offsets[0], j + 0 + in_offsets[1], k + 0 + in_offsets[2]));
  }

  template <class StorageType0>
  static double delta_i_plus_1_interval_start_0_end_0(const int i, const int j, const int k,
                                                      ParamWrapper<gridtools::data_view<StorageType0>> pw_in) {
    gridtools::data_view<StorageType0> in = pw_in.dview_;
    auto in_offsets = pw_in.offsets_;
    return (in(i + 1 + in_offsets[0], j + 0 + in_offsets[1], k + 0 + in_offsets[2]) -
            in(i + 0 + in_offsets[0], j + 0 + in_offsets[1], k + 0 + in_offsets[2]));
  }

  template <class StorageType0>
  static double avg_i_plus_1_interval_start_0_end_0(const int i, const int j, const int k,
                                                    ParamWrapper<gridtools::data_view<StorageType0>> pw_in) {
    gridtools::data_view<StorageType0> in = pw_in.dview_;
    auto in_offsets = pw_in.offsets_;
    return ((gridtools::clang::float_type)0.5 *
            (in(i + 1 + in_offsets[0], j + 0 + in_offsets[1], k + 0 + in_offsets[2]) +
             in(i + 0 + in_offsets[0], j + 0 + in_offsets[1], k + 0 + in_offsets[2])));
  }

  template <class StorageType0>
  static double avg_j_minus_1_interval_start_0_end_0(const int i, const int j, const int k,
                                                     ParamWrapper<gridtools::data_view<StorageType0>> pw_in) {
    gridtools::data_view<StorageType0> in = pw_in.dview_;
    auto in_offsets = pw_in.offsets_;
    return ((gridtools::clang::float_type)0.5 *
            (in(i + 0 + in_offsets[0], j + -1 + in_offsets[1], k + 0 + in_offsets[2]) +
             in(i + 0 + in_offsets[0], j + 0 + in_offsets[1], k + 0 + in_offsets[2])));
  }

  template <class StorageType0>
  static double avg_j_plus_1_interval_start_0_end_0(const int i, const int j, const int k,
                                                    ParamWrapper<gridtools::data_view<StorageType0>> pw_in) {
    gridtools::data_view<StorageType0> in = pw_in.dview_;
    auto in_offsets = pw_in.offsets_;
    return ((gridtools::clang::float_type)0.5 *
            (in(i + 0 + in_offsets[0], j + 1 + in_offsets[1], k + 0 + in_offsets[2]) +
             in(i + 0 + in_offsets[0], j + 0 + in_offsets[1], k + 0 + in_offsets[2])));
  }

  template <class StorageType0>
  static double avg_i_minus_1_interval_start_0_end_0(const int i, const int j, const int k,
                                                     ParamWrapper<gridtools::data_view<StorageType0>> pw_in) {
    gridtools::data_view<StorageType0> in = pw_in.dview_;
    auto in_offsets = pw_in.offsets_;
    return ((gridtools::clang::float_type)0.5 *
            (in(i + -1 + in_offsets[0], j + 0 + in_offsets[1], k + 0 + in_offsets[2]) +
             in(i + 0 + in_offsets[0], j + 0 + in_offsets[1], k + 0 + in_offsets[2])));
  }

  template <class StorageType0, class StorageType1, class StorageType2>
  static double laplacian_interval_start_0_end_0(const int i, const int j, const int k,
                                                 ParamWrapper<gridtools::data_view<StorageType0>> pw_in,
                                                 ParamWrapper<gridtools::data_view<StorageType1>> pw_crlato,
                                                 ParamWrapper<gridtools::data_view<StorageType2>> pw_crlatu) {
    gridtools::data_view<StorageType0> in = pw_in.dview_;
    auto in_offsets = pw_in.offsets_;
    gridtools::data_view<StorageType1> crlato = pw_crlato.dview_;
    auto crlato_offsets = pw_crlato.offsets_;
    gridtools::data_view<StorageType2> crlatu = pw_crlatu.dview_;
    auto crlatu_offsets = pw_crlatu.offsets_;
    return (
        (((in(i + 1 + in_offsets[0], j + 0 + in_offsets[1], k + 0 + in_offsets[2]) +
           in(i + -1 + in_offsets[0], j + 0 + in_offsets[1], k + 0 + in_offsets[2])) -
          ((gridtools::clang::float_type)2 * in(i + 0 + in_offsets[0], j + 0 + in_offsets[1], k + 0 + in_offsets[2]))) +
         (crlato(i + 0 + crlato_offsets[0], j + 0 + crlato_offsets[1], k + 0 + crlato_offsets[2]) *
          delta_j_plus_1_interval_start_0_end_0(
              i, j, k, ParamWrapper<gridtools::data_view<StorageType0>>(in, std::array<int, 3>{0, 0, 0})))) +
        (crlatu(i + 0 + crlatu_offsets[0], j + 0 + crlatu_offsets[1], k + 0 + crlatu_offsets[2]) *
         delta_j_minus_1_interval_start_0_end_0(
             i, j, k, ParamWrapper<gridtools::data_view<StorageType0>>(in, std::array<int, 3>{0, 0, 0}))));
  }

  struct sbase {
    virtual void run() {}

    virtual ~sbase() {}
  };
  template <class StorageType0, class StorageType1, class StorageType2, class StorageType3, class StorageType4,
            class StorageType5, class StorageType6, class StorageType7, class StorageType8, class StorageType9,
            class StorageType10, class StorageType11, class StorageType12, class StorageType13>
  struct stencil_0 : public sbase {
    // //Members
    const gridtools::clang::domain& m_dom;
    StorageType0& m_u_out;
    StorageType1& m_v_out;
    StorageType2& m_u_in;
    StorageType3& m_v_in;
    StorageType4& m_hdmaskvel;
    StorageType5& m_crlavo;
    StorageType6& m_crlavu;
    StorageType7& m_crlato;
    StorageType8& m_crlatu;
    StorageType9& m_acrlat0;
    StorageType10& m_eddlon;
    StorageType11& m_eddlat;
    StorageType12& m_tau_smag;
    StorageType13& m_weight_smag;
    gridtools::clang::meta_data_t m_meta_data;
    gridtools::clang::storage_t m_S_sqr_uv;
    gridtools::clang::storage_t m_T_sqr_s;

   public:
    stencil_0(const gridtools::clang::domain& dom_, StorageType0& u_out_, StorageType1& v_out_, StorageType2& u_in_,
              StorageType3& v_in_, StorageType4& hdmaskvel_, StorageType5& crlavo_, StorageType6& crlavu_,
              StorageType7& crlato_, StorageType8& crlatu_, StorageType9& acrlat0_, StorageType10& eddlon_,
              StorageType11& eddlat_, StorageType12& tau_smag_, StorageType13& weight_smag_)
        : m_dom(dom_),
          m_u_out(u_out_),
          m_v_out(v_out_),
          m_u_in(u_in_),
          m_v_in(v_in_),
          m_hdmaskvel(hdmaskvel_),
          m_crlavo(crlavo_),
          m_crlavu(crlavu_),
          m_crlato(crlato_),
          m_crlatu(crlatu_),
          m_acrlat0(acrlat0_),
          m_eddlon(eddlon_),
          m_eddlat(eddlat_),
          m_tau_smag(tau_smag_),
          m_weight_smag(weight_smag_),
          m_meta_data(dom_.isize(), dom_.jsize(), dom_.ksize()),
          m_S_sqr_uv(m_meta_data),
          m_T_sqr_s(m_meta_data) {}

    ~stencil_0() {}

    virtual void run() {
      gridtools::data_view<StorageType0> u_out = gridtools::make_host_view(m_u_out);
      gridtools::data_view<StorageType1> v_out = gridtools::make_host_view(m_v_out);
      gridtools::data_view<StorageType2> u_in = gridtools::make_host_view(m_u_in);
      gridtools::data_view<StorageType3> v_in = gridtools::make_host_view(m_v_in);
      gridtools::data_view<StorageType4> hdmaskvel = gridtools::make_host_view(m_hdmaskvel);
      gridtools::data_view<StorageType5> crlavo = gridtools::make_host_view(m_crlavo);
      gridtools::data_view<StorageType6> crlavu = gridtools::make_host_view(m_crlavu);
      gridtools::data_view<StorageType7> crlato = gridtools::make_host_view(m_crlato);
      gridtools::data_view<StorageType8> crlatu = gridtools::make_host_view(m_crlatu);
      gridtools::data_view<StorageType9> acrlat0 = gridtools::make_host_view(m_acrlat0);
      gridtools::data_view<StorageType10> eddlon = gridtools::make_host_view(m_eddlon);
      gridtools::data_view<StorageType11> eddlat = gridtools::make_host_view(m_eddlat);
      gridtools::data_view<StorageType12> tau_smag = gridtools::make_host_view(m_tau_smag);
      gridtools::data_view<StorageType13> weight_smag = gridtools::make_host_view(m_weight_smag);
      gridtools::data_view<storage_t> S_sqr_uv = gridtools::make_host_view(m_S_sqr_uv);
      gridtools::data_view<storage_t> T_sqr_s = gridtools::make_host_view(m_T_sqr_s);
      for (int k = 0; k <= (m_dom.ksize() == 0 ? 0 : (m_dom.ksize() - m_dom.kplus() - 1)); ++k) {
        for (int i = m_dom.iminus(); i <= m_dom.isize() - m_dom.iplus() - 1; ++i) {
          for (int j = m_dom.jminus(); j <= m_dom.jsize() - m_dom.jplus() - 1; ++j) {
            const gridtools::clang::float_type __local_frac_1_dx_19 =
                (acrlat0(i + 0, j + 0, k + 0) * eddlon(i + 0, j + 0, k + 0));
            const gridtools::clang::float_type __local_frac_1_dy_20 =
                (eddlat(i + 0, j + 0, k + 0) / (gridtools::clang::float_type)6371229);
            const gridtools::clang::float_type __local_T_s_22 =
                ((delta_j_minus_1_interval_start_0_end_0(
                      i, j, k, ParamWrapper<gridtools::data_view<StorageType3>>(v_in, std::array<int, 3>{0, 0, 0})) *
                  __local_frac_1_dy_20) -
                 (delta_i_minus_1_interval_start_0_end_0(
                      i, j, k, ParamWrapper<gridtools::data_view<StorageType2>>(u_in, std::array<int, 3>{0, 0, 0})) *
                  __local_frac_1_dx_19));
            T_sqr_s(i + 0, j + 0, k + 0) = (__local_T_s_22 * __local_T_s_22);
            const gridtools::clang::float_type __local_S_uv_23 =
                ((delta_j_plus_1_interval_start_0_end_0(
                      i, j, k, ParamWrapper<gridtools::data_view<StorageType2>>(u_in, std::array<int, 3>{0, 0, 0})) *
                  __local_frac_1_dy_20) +
                 (delta_i_plus_1_interval_start_0_end_0(
                      i, j, k, ParamWrapper<gridtools::data_view<StorageType3>>(v_in, std::array<int, 3>{0, 0, 0})) *
                  __local_frac_1_dx_19));
            S_sqr_uv(i + 0, j + 0, k + 0) = (__local_S_uv_23 * __local_S_uv_23);
          }
        }
        for (int i = m_dom.iminus(); i <= m_dom.isize() - m_dom.iplus() - 1; ++i) {
          for (int j = m_dom.jminus(); j <= m_dom.jsize() - m_dom.jplus() - 1; ++j) {
            const gridtools::clang::float_type __local_hdweight_24 =
                (weight_smag(i + 0, j + 0, k + 0) * hdmaskvel(i + 0, j + 0, k + 0));
            gridtools::clang::float_type __local_smag_u_25 =
                ((tau_smag(i + 0, j + 0, k + 0) *
                  gridtools::clang::math::sqrt(
                      (avg_i_plus_1_interval_start_0_end_0(
                           i, j, k, ParamWrapper<gridtools::data_view<gridtools::clang::storage_t>>(
                                        T_sqr_s, std::array<int, 3>{0, 0, 0})) +
                       avg_j_minus_1_interval_start_0_end_0(
                           i, j, k, ParamWrapper<gridtools::data_view<gridtools::clang::storage_t>>(
                                        S_sqr_uv, std::array<int, 3>{0, 0, 0}))))) -
                 __local_hdweight_24);
            __local_smag_u_25 = gridtools::clang::math::min(
                (gridtools::clang::float_type)0.5,
                gridtools::clang::math::max((gridtools::clang::float_type)0, __local_smag_u_25));
            gridtools::clang::float_type __local_smag_v_30 =
                ((tau_smag(i + 0, j + 0, k + 0) *
                  gridtools::clang::math::sqrt(
                      (avg_j_plus_1_interval_start_0_end_0(
                           i, j, k, ParamWrapper<gridtools::data_view<gridtools::clang::storage_t>>(
                                        T_sqr_s, std::array<int, 3>{0, 0, 0})) +
                       avg_i_minus_1_interval_start_0_end_0(
                           i, j, k, ParamWrapper<gridtools::data_view<gridtools::clang::storage_t>>(
                                        S_sqr_uv, std::array<int, 3>{0, 0, 0}))))) -
                 __local_hdweight_24);
            __local_smag_v_30 = gridtools::clang::math::min(
                (gridtools::clang::float_type)0.5,
                gridtools::clang::math::max((gridtools::clang::float_type)0, __local_smag_v_30));
            const gridtools::clang::float_type __local_lapu_35 = laplacian_interval_start_0_end_0(
                i, j, k, ParamWrapper<gridtools::data_view<StorageType2>>(u_in, std::array<int, 3>{0, 0, 0}),
                ParamWrapper<gridtools::data_view<StorageType7>>(crlato, std::array<int, 3>{0, 0, 0}),
                ParamWrapper<gridtools::data_view<StorageType8>>(crlatu, std::array<int, 3>{0, 0, 0}));
            const gridtools::clang::float_type __local_lapv_37 = laplacian_interval_start_0_end_0(
                i, j, k, ParamWrapper<gridtools::data_view<StorageType3>>(v_in, std::array<int, 3>{0, 0, 0}),
                ParamWrapper<gridtools::data_view<StorageType5>>(crlavo, std::array<int, 3>{0, 0, 0}),
                ParamWrapper<gridtools::data_view<StorageType6>>(crlavu, std::array<int, 3>{0, 0, 0}));
            u_out(i + 0, j + 0, k + 0) = (u_in(i + 0, j + 0, k + 0) + (__local_smag_u_25 * __local_lapu_35));
            v_out(i + 0, j + 0, k + 0) = (v_in(i + 0, j + 0, k + 0) + (__local_smag_v_30 * __local_lapv_37));
          }
        }
      }
    }
  };
  static constexpr const char* s_name = "hd_smagorinsky_stencil";
  sbase* m_stencil_0;

 public:
  hd_smagorinsky_stencil(const hd_smagorinsky_stencil&) = delete;

  // Members

  template <class StorageType1, class StorageType2, class StorageType3, class StorageType4, class StorageType5,
            class StorageType6, class StorageType7, class StorageType8, class StorageType9, class StorageType10,
            class StorageType11, class StorageType12, class StorageType13, class StorageType14>
  hd_smagorinsky_stencil(const gridtools::clang::domain& dom, StorageType1& u_out, StorageType2& v_out,
                         StorageType3& u_in, StorageType4& v_in, StorageType5& hdmaskvel, StorageType6& crlavo,
                         StorageType7& crlavu, StorageType8& crlato, StorageType9& crlatu, StorageType10& acrlat0,
                         StorageType11& eddlon, StorageType12& eddlat, StorageType13& tau_smag,
                         StorageType14& weight_smag)
      : m_stencil_0(new stencil_0<StorageType1, StorageType2, StorageType3, StorageType4, StorageType5, StorageType6,
                                  StorageType7, StorageType8, StorageType9, StorageType10, StorageType11, StorageType12,
                                  StorageType13, StorageType14>(dom, u_out, v_out, u_in, v_in, hdmaskvel, crlavo,
                                                                crlavu, crlato, crlatu, acrlat0, eddlon, eddlat,
                                                                tau_smag, weight_smag)) {}

  void run() { m_stencil_0->run(); }
};
}  // namespace cxxnaiv
;
