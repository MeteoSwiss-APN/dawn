// gtclang (0.0.1-b50903a-x86_64-linux-gnu-5.4.0)
// based on LLVM/Clang (3.8.0), Dawn (0.0.1)
// Generated on 2018-01-04  20:30:25

#define GRIDTOOLS_CLANG_GENERATED 1
#ifndef BOOST_RESULT_OF_USE_TR1
 #define BOOST_RESULT_OF_USE_TR1 1
#endif
#ifndef BOOST_NO_CXX11_DECLTYPE
 #define BOOST_NO_CXX11_DECLTYPE 1
#endif
#ifndef FUSION_MAX_VECTOR_SIZE
 #define FUSION_MAX_VECTOR_SIZE 20
#endif
#ifndef FUSION_MAX_MAP_SIZE
 #define FUSION_MAX_MAP_SIZE 20
#endif
#ifndef BOOST_MPL_LIMIT_VECTOR_SIZE
 #define BOOST_MPL_LIMIT_VECTOR_SIZE 20
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

namespace gridtools {

class hd_smagorinsky_stencil {
 public:
  struct stencil_0 {
    // Intervals
    using interval_start_0_end_0 = gridtools::interval<gridtools::level<0, -1>, gridtools::level<1, -1>>;
    using axis_stencil_0 = gridtools::interval<gridtools::level<0, -2>, gridtools::level<1, 1>>;
    using grid_stencil_0 = gridtools::grid<axis_stencil_0>;

    // Members
    std::shared_ptr<gridtools::stencil> m_stencil;
    std::unique_ptr<grid_stencil_0> m_grid;

    struct delta_j_minus_1_interval_start_0_end_0 {
      using __out = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using in = gridtools::accessor<1, gridtools::enumtype::in, gridtools::extent<0, 0, -1, 0, 0, 0>>;
      using arg_list = boost::mpl::vector<__out, in>;

      template <typename Evaluation>
      GT_FUNCTION static void Do(Evaluation& eval, interval_start_0_end_0) {
        eval(__out(0, 0, 0)) = (eval(in(0, -1, 0)) - eval(in(0, 0, 0)));
      }
    };

    struct delta_i_minus_1_interval_start_0_end_0 {
      using __out = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using in = gridtools::accessor<1, gridtools::enumtype::in, gridtools::extent<-1, 0, 0, 0, 0, 0>>;
      using arg_list = boost::mpl::vector<__out, in>;

      template <typename Evaluation>
      GT_FUNCTION static void Do(Evaluation& eval, interval_start_0_end_0) {
        eval(__out(0, 0, 0)) = (eval(in(-1, 0, 0)) - eval(in(0, 0, 0)));
      }
    };

    struct delta_j_plus_1_interval_start_0_end_0 {
      using __out = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using in = gridtools::accessor<1, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 1, 0, 0>>;
      using arg_list = boost::mpl::vector<__out, in>;

      template <typename Evaluation>
      GT_FUNCTION static void Do(Evaluation& eval, interval_start_0_end_0) {
        eval(__out(0, 0, 0)) = (eval(in(0, 1, 0)) - eval(in(0, 0, 0)));
      }
    };

    struct delta_i_plus_1_interval_start_0_end_0 {
      using __out = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using in = gridtools::accessor<1, gridtools::enumtype::in, gridtools::extent<0, 1, 0, 0, 0, 0>>;
      using arg_list = boost::mpl::vector<__out, in>;

      template <typename Evaluation>
      GT_FUNCTION static void Do(Evaluation& eval, interval_start_0_end_0) {
        eval(__out(0, 0, 0)) = (eval(in(1, 0, 0)) - eval(in(0, 0, 0)));
      }
    };

    struct avg_i_plus_1_interval_start_0_end_0 {
      using __out = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using in = gridtools::accessor<1, gridtools::enumtype::in, gridtools::extent<0, 1, 0, 0, 0, 0>>;
      using arg_list = boost::mpl::vector<__out, in>;

      template <typename Evaluation>
      GT_FUNCTION static void Do(Evaluation& eval, interval_start_0_end_0) {
        eval(__out(0, 0, 0)) = ((gridtools::clang::float_type)0.5 * (eval(in(1, 0, 0)) + eval(in(0, 0, 0))));
      }
    };

    struct avg_j_minus_1_interval_start_0_end_0 {
      using __out = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using in = gridtools::accessor<1, gridtools::enumtype::in, gridtools::extent<0, 0, -1, 0, 0, 0>>;
      using arg_list = boost::mpl::vector<__out, in>;

      template <typename Evaluation>
      GT_FUNCTION static void Do(Evaluation& eval, interval_start_0_end_0) {
        eval(__out(0, 0, 0)) = ((gridtools::clang::float_type)0.5 * (eval(in(0, -1, 0)) + eval(in(0, 0, 0))));
      }
    };

    struct avg_j_plus_1_interval_start_0_end_0 {
      using __out = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using in = gridtools::accessor<1, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 1, 0, 0>>;
      using arg_list = boost::mpl::vector<__out, in>;

      template <typename Evaluation>
      GT_FUNCTION static void Do(Evaluation& eval, interval_start_0_end_0) {
        eval(__out(0, 0, 0)) = ((gridtools::clang::float_type)0.5 * (eval(in(0, 1, 0)) + eval(in(0, 0, 0))));
      }
    };

    struct avg_i_minus_1_interval_start_0_end_0 {
      using __out = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using in = gridtools::accessor<1, gridtools::enumtype::in, gridtools::extent<-1, 0, 0, 0, 0, 0>>;
      using arg_list = boost::mpl::vector<__out, in>;

      template <typename Evaluation>
      GT_FUNCTION static void Do(Evaluation& eval, interval_start_0_end_0) {
        eval(__out(0, 0, 0)) = ((gridtools::clang::float_type)0.5 * (eval(in(-1, 0, 0)) + eval(in(0, 0, 0))));
      }
    };

    struct laplacian_interval_start_0_end_0 {
      using __out = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using in = gridtools::accessor<1, gridtools::enumtype::in, gridtools::extent<-1, 1, -1, 1, 0, 0>>;
      using crlato = gridtools::accessor<2, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using crlatu = gridtools::accessor<3, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using arg_list = boost::mpl::vector<__out, in, crlato, crlatu>;

      template <typename Evaluation>
      GT_FUNCTION static void Do(Evaluation& eval, interval_start_0_end_0) {
        eval(__out(0, 0, 0)) =
            ((((eval(in(1, 0, 0)) + eval(in(-1, 0, 0))) - ((gridtools::clang::float_type)2 * eval(in(0, 0, 0)))) +
              (eval(crlato(0, 0, 0)) *
               gridtools::call<delta_j_plus_1_interval_start_0_end_0, interval_start_0_end_0>::with(eval,
                                                                                                    in(0, 0, 0)))) +
             (eval(crlatu(0, 0, 0)) *
              gridtools::call<delta_j_minus_1_interval_start_0_end_0, interval_start_0_end_0>::with(eval,
                                                                                                    in(0, 0, 0))));
      }
    };

    struct stage_0_0 {
      using T_sqr_s = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using S_sqr_uv = gridtools::accessor<1, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using u_in = gridtools::accessor<2, gridtools::enumtype::in, gridtools::extent<-1, 0, 0, 1, 0, 0>>;
      using v_in = gridtools::accessor<3, gridtools::enumtype::in, gridtools::extent<0, 1, -1, 0, 0, 0>>;
      using acrlat0 = gridtools::accessor<4, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using eddlon = gridtools::accessor<5, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using eddlat = gridtools::accessor<6, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using arg_list = boost::mpl::vector<T_sqr_s, S_sqr_uv, u_in, v_in, acrlat0, eddlon, eddlat>;

      template <typename Evaluation>
      GT_FUNCTION static void Do(Evaluation& eval, interval_start_0_end_0) {
        const gridtools::clang::float_type __local_frac_1_dx_19 = (eval(acrlat0(0, 0, 0)) * eval(eddlon(0, 0, 0)));
        const gridtools::clang::float_type __local_frac_1_dy_20 =
            (eval(eddlat(0, 0, 0)) / (gridtools::clang::float_type)6371229);
        const gridtools::clang::float_type __local_T_s_22 =
            ((gridtools::call<delta_j_minus_1_interval_start_0_end_0, interval_start_0_end_0>::with(eval,
                                                                                                    v_in(0, 0, 0)) *
              __local_frac_1_dy_20) -
             (gridtools::call<delta_i_minus_1_interval_start_0_end_0, interval_start_0_end_0>::with(eval,
                                                                                                    u_in(0, 0, 0)) *
              __local_frac_1_dx_19));
        eval(T_sqr_s(0, 0, 0)) = (__local_T_s_22 * __local_T_s_22);
        const gridtools::clang::float_type __local_S_uv_23 =
            ((gridtools::call<delta_j_plus_1_interval_start_0_end_0, interval_start_0_end_0>::with(eval,
                                                                                                   u_in(0, 0, 0)) *
              __local_frac_1_dy_20) +
             (gridtools::call<delta_i_plus_1_interval_start_0_end_0, interval_start_0_end_0>::with(eval,
                                                                                                   v_in(0, 0, 0)) *
              __local_frac_1_dx_19));
        eval(S_sqr_uv(0, 0, 0)) = (__local_S_uv_23 * __local_S_uv_23);
      }
    };

    struct stage_0_1 {
      using u_out = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using v_out = gridtools::accessor<1, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using u_in = gridtools::accessor<2, gridtools::enumtype::in, gridtools::extent<-1, 1, -1, 1, 0, 0>>;
      using v_in = gridtools::accessor<3, gridtools::enumtype::in, gridtools::extent<-1, 1, -1, 1, 0, 0>>;
      using hdmaskvel = gridtools::accessor<4, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using crlavo = gridtools::accessor<5, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using crlavu = gridtools::accessor<6, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using crlato = gridtools::accessor<7, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using crlatu = gridtools::accessor<8, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using tau_smag = gridtools::accessor<9, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using weight_smag = gridtools::accessor<10, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using T_sqr_s = gridtools::accessor<11, gridtools::enumtype::in, gridtools::extent<0, 1, 0, 1, 0, 0>>;
      using S_sqr_uv = gridtools::accessor<12, gridtools::enumtype::in, gridtools::extent<-1, 0, -1, 0, 0, 0>>;
      using arg_list = boost::mpl::vector<u_out, v_out, u_in, v_in, hdmaskvel, crlavo, crlavu, crlato, crlatu, tau_smag,
                                          weight_smag, T_sqr_s, S_sqr_uv>;

      template <typename Evaluation>
      GT_FUNCTION static void Do(Evaluation& eval, interval_start_0_end_0) {
        const gridtools::clang::float_type __local_hdweight_24 =
            (eval(weight_smag(0, 0, 0)) * eval(hdmaskvel(0, 0, 0)));
        gridtools::clang::float_type __local_smag_u_25 =
            ((eval(tau_smag(0, 0, 0)) *
              gridtools::clang::math::sqrt(
                  (gridtools::call<avg_i_plus_1_interval_start_0_end_0, interval_start_0_end_0>::with(
                       eval, T_sqr_s(0, 0, 0)) +
                   gridtools::call<avg_j_minus_1_interval_start_0_end_0, interval_start_0_end_0>::with(
                       eval, S_sqr_uv(0, 0, 0))))) -
             __local_hdweight_24);
        __local_smag_u_25 = gridtools::clang::math::min(
            (gridtools::clang::float_type)0.5,
            gridtools::clang::math::max((gridtools::clang::float_type)0, __local_smag_u_25));
        gridtools::clang::float_type __local_smag_v_30 =
            ((eval(tau_smag(0, 0, 0)) *
              gridtools::clang::math::sqrt(
                  (gridtools::call<avg_j_plus_1_interval_start_0_end_0, interval_start_0_end_0>::with(
                       eval, T_sqr_s(0, 0, 0)) +
                   gridtools::call<avg_i_minus_1_interval_start_0_end_0, interval_start_0_end_0>::with(
                       eval, S_sqr_uv(0, 0, 0))))) -
             __local_hdweight_24);
        __local_smag_v_30 = gridtools::clang::math::min(
            (gridtools::clang::float_type)0.5,
            gridtools::clang::math::max((gridtools::clang::float_type)0, __local_smag_v_30));
        const gridtools::clang::float_type __local_lapu_35 =
            gridtools::call<laplacian_interval_start_0_end_0, interval_start_0_end_0>::with(
                eval, u_in(0, 0, 0), crlato(0, 0, 0), crlatu(0, 0, 0));
        const gridtools::clang::float_type __local_lapv_37 =
            gridtools::call<laplacian_interval_start_0_end_0, interval_start_0_end_0>::with(
                eval, v_in(0, 0, 0), crlavo(0, 0, 0), crlavu(0, 0, 0));
        eval(u_out(0, 0, 0)) = (eval(u_in(0, 0, 0)) + (__local_smag_u_25 * __local_lapu_35));
        eval(v_out(0, 0, 0)) = (eval(v_in(0, 0, 0)) + (__local_smag_v_30 * __local_lapv_37));
      }
    };

    template <class S1, class S2, class S3, class S4, class S5, class S6, class S7, class S8, class S9, class S10,
              class S11, class S12, class S13, class S14>
    stencil_0(const gridtools::clang::domain& dom, S1& u_out, S2& v_out, S3& u_in, S4& v_in, S5& hdmaskvel, S6& crlavo,
              S7& crlavu, S8& crlato, S9& crlatu, S10& acrlat0, S11& eddlon, S12& eddlat, S13& tau_smag,
              S14& weight_smag) {
      // Domain
      using p_S_sqr_uv = gridtools::tmp_arg<0, storage_t>;
      using p_T_sqr_s = gridtools::tmp_arg<1, storage_t>;
      using p_u_out = gridtools::arg<2, S1>;
      using p_v_out = gridtools::arg<3, S2>;
      using p_u_in = gridtools::arg<4, S3>;
      using p_v_in = gridtools::arg<5, S4>;
      using p_hdmaskvel = gridtools::arg<6, S5>;
      using p_crlavo = gridtools::arg<7, S6>;
      using p_crlavu = gridtools::arg<8, S7>;
      using p_crlato = gridtools::arg<9, S8>;
      using p_crlatu = gridtools::arg<10, S9>;
      using p_acrlat0 = gridtools::arg<11, S10>;
      using p_eddlon = gridtools::arg<12, S11>;
      using p_eddlat = gridtools::arg<13, S12>;
      using p_tau_smag = gridtools::arg<14, S13>;
      using p_weight_smag = gridtools::arg<15, S14>;
      using domain_arg_list =
          boost::mpl::vector<p_S_sqr_uv, p_T_sqr_s, p_u_out, p_v_out, p_u_in, p_v_in, p_hdmaskvel, p_crlavo, p_crlavu,
                             p_crlato, p_crlatu, p_acrlat0, p_eddlon, p_eddlat, p_tau_smag, p_weight_smag>;
      auto gt_domain =
          new gridtools::aggregator_type<domain_arg_list>(u_out, v_out, u_in, v_in, hdmaskvel, crlavo, crlavu, crlato,
                                                          crlatu, acrlat0, eddlon, eddlat, tau_smag, weight_smag);

      // Grid
      unsigned int di[5] = {dom.iminus(), dom.iminus(), dom.iplus(), dom.isize() - 1 - dom.iplus(), dom.isize()};
      unsigned int dj[5] = {dom.jminus(), dom.jminus(), dom.jplus(), dom.jsize() - 1 - dom.jplus(), dom.jsize()};
      m_grid = std::unique_ptr<grid_stencil_0>(new grid_stencil_0(di, dj));
      m_grid->value_list[0] = dom.kminus();
      m_grid->value_list[1] = dom.ksize() == 0 ? 0 : dom.ksize() - dom.kplus() - 1;

      // Computation
      m_stencil = gridtools::make_computation<gridtools::clang::backend_t>(
          *gt_domain, *m_grid,
          gridtools::make_multistage(
              gridtools::enumtype::execute<gridtools::enumtype::forward /*parallel*/>(),
              gridtools::define_caches(gridtools::cache<gridtools::IJ, gridtools::cache_io_policy::local>(p_S_sqr_uv()),
                                       gridtools::cache<gridtools::IJ, gridtools::cache_io_policy::local>(p_T_sqr_s())),
              gridtools::make_stage<stage_0_0>(p_T_sqr_s(), p_S_sqr_uv(), p_u_in(), p_v_in(), p_acrlat0(), p_eddlon(),
                                               p_eddlat()),
              gridtools::make_stage<stage_0_1>(p_u_out(), p_v_out(), p_u_in(), p_v_in(), p_hdmaskvel(), p_crlavo(),
                                               p_crlavu(), p_crlato(), p_crlatu(), p_tau_smag(), p_weight_smag(),
                                               p_T_sqr_s(), p_S_sqr_uv())));
    }

    ~stencil_0() { m_stencil->finalize(); }

    gridtools::stencil* get_stencil() { return m_stencil.get(); }
  };

  // Members
  stencil_0 m_stencil_0;
  static constexpr const char* s_name = "hd_smagorinsky_stencil";

 public:
  hd_smagorinsky_stencil(const hd_smagorinsky_stencil&) = delete;

  template <class S1, class S2, class S3, class S4, class S5, class S6, class S7, class S8, class S9, class S10,
            class S11, class S12, class S13, class S14>
  hd_smagorinsky_stencil(const gridtools::clang::domain& dom, S1& u_out, S2& v_out, S3& u_in, S4& v_in, S5& hdmaskvel,
                         S6& crlavo, S7& crlavu, S8& crlato, S9& crlatu, S10& acrlat0, S11& eddlon, S12& eddlat,
                         S13& tau_smag, S14& weight_smag)
      : m_stencil_0(dom, u_out, v_out, u_in, v_in, hdmaskvel, crlavo, crlavu, crlato, crlatu, acrlat0, eddlon, eddlat,
                    tau_smag, weight_smag) {
    static_assert(gridtools::is_data_store<S1>::value,
                  "argument 'u_out' is not a 'gridtools::data_store' (2nd argument invalid)");
    static_assert(gridtools::is_data_store<S2>::value,
                  "argument 'v_out' is not a 'gridtools::data_store' (3rd argument invalid)");
    static_assert(gridtools::is_data_store<S3>::value,
                  "argument 'u_in' is not a 'gridtools::data_store' (4th argument invalid)");
    static_assert(gridtools::is_data_store<S4>::value,
                  "argument 'v_in' is not a 'gridtools::data_store' (5th argument invalid)");
    static_assert(gridtools::is_data_store<S5>::value,
                  "argument 'hdmaskvel' is not a 'gridtools::data_store' (6th argument invalid)");
    static_assert(gridtools::is_data_store<S6>::value,
                  "argument 'crlavo' is not a 'gridtools::data_store' (7th argument invalid)");
    static_assert(gridtools::is_data_store<S7>::value,
                  "argument 'crlavu' is not a 'gridtools::data_store' (8th argument invalid)");
    static_assert(gridtools::is_data_store<S8>::value,
                  "argument 'crlato' is not a 'gridtools::data_store' (9th argument invalid)");
    static_assert(gridtools::is_data_store<S9>::value,
                  "argument 'crlatu' is not a 'gridtools::data_store' (10th argument invalid)");
    static_assert(gridtools::is_data_store<S10>::value,
                  "argument 'acrlat0' is not a 'gridtools::data_store' (11th argument invalid)");
    static_assert(gridtools::is_data_store<S11>::value,
                  "argument 'eddlon' is not a 'gridtools::data_store' (12th argument invalid)");
    static_assert(gridtools::is_data_store<S12>::value,
                  "argument 'eddlat' is not a 'gridtools::data_store' (13th argument invalid)");
    static_assert(gridtools::is_data_store<S13>::value,
                  "argument 'tau_smag' is not a 'gridtools::data_store' (14th argument invalid)");
    static_assert(gridtools::is_data_store<S14>::value,
                  "argument 'weight_smag' is not a 'gridtools::data_store' (15th argument invalid)");
  }

  void make_steady() {
    m_stencil_0.get_stencil()->ready();
    m_stencil_0.get_stencil()->steady();
  }

  void run(bool make_steady = true) {
    if (make_steady) this->make_steady();
    m_stencil_0.get_stencil()->run();
  }

  std::vector<gridtools::stencil*> get_stencils() {
    return std::vector<gridtools::stencil*>({m_stencil_0.get_stencil()});
  }

  const char* get_name() const { return s_name; }
};
}  // namespace gridtool
;
