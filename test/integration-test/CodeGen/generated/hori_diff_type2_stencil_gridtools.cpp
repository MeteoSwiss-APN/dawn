// gtclang (0.0.1-b9691ca-x86_64-linux-gnu-5.4.0)
// based on LLVM/Clang (3.8.0), Dawn (0.0.1)
// Generated on 2018-01-02  19:43:46

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
#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

;

;

;

;

namespace gridtools {

class hori_diff_type2_stencil {
 public:
  struct stencil_0 {
    // Intervals
    using interval_start_0_end_0 = gridtools::interval<gridtools::level<0, -1>, gridtools::level<1, -1>>;
    using axis_stencil_0 = gridtools::interval<gridtools::level<0, -2>, gridtools::level<1, 1>>;
    using grid_stencil_0 = gridtools::grid<axis_stencil_0>;

    // Members
    std::shared_ptr<gridtools::stencil> m_stencil;
    std::unique_ptr<grid_stencil_0> m_grid;

    struct laplacian_interval_start_0_end_0 {
      using __out = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using data = gridtools::accessor<1, gridtools::enumtype::in, gridtools::extent<-1, 1, -1, 1, 0, 0>>;
      using crlato = gridtools::accessor<2, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using crlatu = gridtools::accessor<3, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using arg_list = boost::mpl::vector<__out, data, crlato, crlatu>;

      template <typename Evaluation>
      GT_FUNCTION static void Do(Evaluation& eval, interval_start_0_end_0) {
        eval(__out(0, 0, 0)) =
            ((((eval(data(1, 0, 0)) + eval(data(-1, 0, 0))) - ((gridtools::clang::float_type)2 * eval(data(0, 0, 0)))) +
              (eval(crlato(0, 0, 0)) *
               gridtools::call<delta_j_plus_1_interval_start_0_end_0, interval_start_0_end_0>::with(eval,
                                                                                                    data(0, 0, 0)))) +
             (eval(crlatu(0, 0, 0)) *
              gridtools::call<delta_j_minus_1_interval_start_0_end_0, interval_start_0_end_0>::with(eval,
                                                                                                    data(0, 0, 0))));
      }
    };

    struct delta_j_plus_1_interval_start_0_end_0 {
      using __out = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using data = gridtools::accessor<1, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 1, 0, 0>>;
      using arg_list = boost::mpl::vector<__out, data>;

      template <typename Evaluation>
      GT_FUNCTION static void Do(Evaluation& eval, interval_start_0_end_0) {
        eval(__out(0, 0, 0)) = (eval(data(0, 1, 0)) - eval(data(0, 0, 0)));
      }
    };

    struct delta_j_minus_1_interval_start_0_end_0 {
      using __out = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using data = gridtools::accessor<1, gridtools::enumtype::in, gridtools::extent<0, 0, -1, 0, 0, 0>>;
      using arg_list = boost::mpl::vector<__out, data>;

      template <typename Evaluation>
      GT_FUNCTION static void Do(Evaluation& eval, interval_start_0_end_0) {
        eval(__out(0, 0, 0)) = (eval(data(0, -1, 0)) - eval(data(0, 0, 0)));
      }
    };

    struct diffusive_flux_x_interval_start_0_end_0 {
      using __out = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using lap = gridtools::accessor<1, gridtools::enumtype::in, gridtools::extent<0, 1, 0, 0, 0, 0>>;
      using data = gridtools::accessor<2, gridtools::enumtype::in, gridtools::extent<0, 1, 0, 0, 0, 0>>;
      using arg_list = boost::mpl::vector<__out, lap, data>;

      template <typename Evaluation>
      GT_FUNCTION static void Do(Evaluation& eval, interval_start_0_end_0) {
        const gridtools::clang::float_type __local_flx_11 =
            gridtools::call<delta_i_plus_1_interval_start_0_end_0, interval_start_0_end_0>::with(eval, lap(0, 0, 0));
        eval(__out(0, 0, 0)) =
            (((__local_flx_11 * gridtools::call<delta_i_plus_1_interval_start_0_end_0, interval_start_0_end_0>::with(
                                    eval, data(0, 0, 0))) > (gridtools::clang::float_type)0)
                 ? (gridtools::clang::float_type)0
                 : __local_flx_11);
      }
    };

    struct delta_i_plus_1_interval_start_0_end_0 {
      using __out = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using data = gridtools::accessor<1, gridtools::enumtype::in, gridtools::extent<0, 1, 0, 0, 0, 0>>;
      using arg_list = boost::mpl::vector<__out, data>;

      template <typename Evaluation>
      GT_FUNCTION static void Do(Evaluation& eval, interval_start_0_end_0) {
        eval(__out(0, 0, 0)) = (eval(data(1, 0, 0)) - eval(data(0, 0, 0)));
      }
    };

    struct diffusive_flux_y_interval_start_0_end_0 {
      using __out = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using lap = gridtools::accessor<1, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 1, 0, 0>>;
      using data = gridtools::accessor<2, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 1, 0, 0>>;
      using crlato = gridtools::accessor<3, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using arg_list = boost::mpl::vector<__out, lap, data, crlato>;

      template <typename Evaluation>
      GT_FUNCTION static void Do(Evaluation& eval, interval_start_0_end_0) {
        const gridtools::clang::float_type __local_fly_18 =
            (eval(crlato(0, 0, 0)) *
             gridtools::call<delta_j_plus_1_interval_start_0_end_0, interval_start_0_end_0>::with(eval, lap(0, 0, 0)));
        eval(__out(0, 0, 0)) =
            (((__local_fly_18 * gridtools::call<delta_j_plus_1_interval_start_0_end_0, interval_start_0_end_0>::with(
                                    eval, data(0, 0, 0))) > (gridtools::clang::float_type)0)
                 ? (gridtools::clang::float_type)0
                 : __local_fly_18);
      }
    };

    struct stage_0_0 {
      using lap = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using in = gridtools::accessor<1, gridtools::enumtype::in, gridtools::extent<-1, 1, -1, 1, 0, 0>>;
      using crlato = gridtools::accessor<2, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using crlatu = gridtools::accessor<3, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using arg_list = boost::mpl::vector<lap, in, crlato, crlatu>;

      template <typename Evaluation>
      GT_FUNCTION static void Do(Evaluation& eval, interval_start_0_end_0) {
        eval(lap(0, 0, 0)) = gridtools::call<laplacian_interval_start_0_end_0, interval_start_0_end_0>::with(
            eval, in(0, 0, 0), crlato(0, 0, 0), crlatu(0, 0, 0));
      }
    };

    struct stage_0_1 {
      using out = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using in = gridtools::accessor<1, gridtools::enumtype::in, gridtools::extent<-1, 1, -1, 1, 0, 0>>;
      using crlato = gridtools::accessor<2, gridtools::enumtype::in, gridtools::extent<0, 0, -1, 0, 0, 0>>;
      using hdmask = gridtools::accessor<3, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using lap = gridtools::accessor<4, gridtools::enumtype::in, gridtools::extent<-1, 1, -1, 1, 0, 0>>;
      using arg_list = boost::mpl::vector<out, in, crlato, hdmask, lap>;

      template <typename Evaluation>
      GT_FUNCTION static void Do(Evaluation& eval, interval_start_0_end_0) {
        const gridtools::clang::float_type __local_delta_flux_x_10 =
            (gridtools::call<diffusive_flux_x_interval_start_0_end_0, interval_start_0_end_0>::with(eval, lap(0, 0, 0),
                                                                                                    in(0, 0, 0)) -
             gridtools::call<diffusive_flux_x_interval_start_0_end_0, interval_start_0_end_0>::with(eval, lap(-1, 0, 0),
                                                                                                    in(-1, 0, 0)));
        const gridtools::clang::float_type __local_delta_flux_y_17 =
            (gridtools::call<diffusive_flux_y_interval_start_0_end_0, interval_start_0_end_0>::with(
                 eval, lap(0, 0, 0), in(0, 0, 0), crlato(0, 0, 0)) -
             gridtools::call<diffusive_flux_y_interval_start_0_end_0, interval_start_0_end_0>::with(
                 eval, lap(0, -1, 0), in(0, -1, 0), crlato(0, -1, 0)));
        eval(out(0, 0, 0)) =
            (eval(in(0, 0, 0)) - (eval(hdmask(0, 0, 0)) * (__local_delta_flux_x_10 + __local_delta_flux_y_17)));
      }
    };

    template <class S1, class S2, class S3, class S4, class S5>
    stencil_0(const gridtools::clang::domain& dom, S1& out, S2& in, S3& crlato, S4& crlatu, S5& hdmask) {
      // Domain
      using p_lap = gridtools::tmp_arg<0, storage_t>;
      using p_out = gridtools::arg<1, S1>;
      using p_in = gridtools::arg<2, S2>;
      using p_crlato = gridtools::arg<3, S3>;
      using p_crlatu = gridtools::arg<4, S4>;
      using p_hdmask = gridtools::arg<5, S5>;
      using domain_arg_list = boost::mpl::vector<p_lap, p_out, p_in, p_crlato, p_crlatu, p_hdmask>;
      auto gt_domain = new gridtools::aggregator_type<domain_arg_list>(out, in, crlato, crlatu, hdmask);

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
              gridtools::define_caches(gridtools::cache<gridtools::IJ, gridtools::cache_io_policy::local>(p_lap())),
              gridtools::make_stage<stage_0_0>(p_lap(), p_in(), p_crlato(), p_crlatu()),
              gridtools::make_stage<stage_0_1>(p_out(), p_in(), p_crlato(), p_hdmask(), p_lap())));
    }

    ~stencil_0() { m_stencil->finalize(); }

    gridtools::stencil* get_stencil() { return m_stencil.get(); }
  };

  // Members
  stencil_0 m_stencil_0;
  static constexpr const char* s_name = "hori_diff_type2_stencil";

 public:
  hori_diff_type2_stencil(const hori_diff_type2_stencil&) = delete;

  template <class S1, class S2, class S3, class S4, class S5>
  hori_diff_type2_stencil(const gridtools::clang::domain& dom, S1& out, S2& in, S3& crlato, S4& crlatu, S5& hdmask)
      : m_stencil_0(dom, out, in, crlato, crlatu, hdmask) {
    static_assert(gridtools::is_data_store<S1>::value,
                  "argument 'out' is not a 'gridtools::data_store' (2nd argument invalid)");
    static_assert(gridtools::is_data_store<S2>::value,
                  "argument 'in' is not a 'gridtools::data_store' (3rd argument invalid)");
    static_assert(gridtools::is_data_store<S3>::value,
                  "argument 'crlato' is not a 'gridtools::data_store' (4th argument invalid)");
    static_assert(gridtools::is_data_store<S4>::value,
                  "argument 'crlatu' is not a 'gridtools::data_store' (5th argument invalid)");
    static_assert(gridtools::is_data_store<S5>::value,
                  "argument 'hdmask' is not a 'gridtools::data_store' (6th argument invalid)");
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
