// gtclang (0.0.1-6e50cdb-x86_64-linux-gnu-5.4.0)
// based on LLVM/Clang (3.8.0), Dawn (0.0.1)
// Generated on 2018-01-04  10:33:53

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

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

namespace gridtools {

class coriolis_stencil {
 public:
  struct stencil_0 {
    // Intervals
    using interval_start_0_end_0 = gridtools::interval<gridtools::level<0, -1>, gridtools::level<1, -1>>;
    using axis_stencil_0 = gridtools::interval<gridtools::level<0, -2>, gridtools::level<1, 1>>;
    using grid_stencil_0 = gridtools::grid<axis_stencil_0>;

    // Members
    std::shared_ptr<gridtools::stencil> m_stencil;
    std::unique_ptr<grid_stencil_0> m_grid;

    struct stage_0_0 {
      using u_tens = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using v_tens = gridtools::accessor<1, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using u_nnow = gridtools::accessor<2, gridtools::enumtype::in, gridtools::extent<-1, 0, 0, 1, 0, 0>>;
      using v_nnow = gridtools::accessor<3, gridtools::enumtype::in, gridtools::extent<0, 1, -1, 0, 0, 0>>;
      using fc = gridtools::accessor<4, gridtools::enumtype::in, gridtools::extent<-1, 0, -1, 0, 0, 0>>;
      using arg_list = boost::mpl::vector<u_tens, v_tens, u_nnow, v_nnow, fc>;

      template <typename Evaluation>
      GT_FUNCTION static void Do(Evaluation& eval, interval_start_0_end_0) {
        gridtools::clang::float_type __local_z_fv_north_8 =
            (eval(fc(0, 0, 0)) * (eval(v_nnow(0, 0, 0)) + eval(v_nnow(1, 0, 0))));
        gridtools::clang::float_type __local_z_fv_south_9 =
            (eval(fc(0, -1, 0)) * (eval(v_nnow(0, -1, 0)) + eval(v_nnow(1, -1, 0))));
        eval(u_tens(0, 0, 0)) += ((gridtools::clang::float_type)0.25 * (__local_z_fv_north_8 + __local_z_fv_south_9));
        gridtools::clang::float_type __local_z_fu_east_11 =
            (eval(fc(0, 0, 0)) * (eval(u_nnow(0, 0, 0)) + eval(u_nnow(0, 1, 0))));
        gridtools::clang::float_type __local_z_fu_west_12 =
            (eval(fc(-1, 0, 0)) * (eval(u_nnow(-1, 0, 0)) + eval(u_nnow(-1, 1, 0))));
        eval(v_tens(0, 0, 0)) -= ((gridtools::clang::float_type)0.25 * (__local_z_fu_east_11 + __local_z_fu_west_12));
      }
    };

    template <class S1, class S2, class S3, class S4, class S5>
    stencil_0(const gridtools::clang::domain& dom, S1& u_tens, S2& u_nnow, S3& v_tens, S4& v_nnow, S5& fc) {
      // Domain
      using p_u_tens = gridtools::arg<0, S1>;
      using p_u_nnow = gridtools::arg<1, S2>;
      using p_v_tens = gridtools::arg<2, S3>;
      using p_v_nnow = gridtools::arg<3, S4>;
      using p_fc = gridtools::arg<4, S5>;
      using domain_arg_list = boost::mpl::vector<p_u_tens, p_u_nnow, p_v_tens, p_v_nnow, p_fc>;
      auto gt_domain = new gridtools::aggregator_type<domain_arg_list>(u_tens, u_nnow, v_tens, v_nnow, fc);

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
              gridtools::define_caches(
                  gridtools::cache<gridtools::K, gridtools::cache_io_policy::fill_and_flush, interval_start_0_end_0>(
                      p_v_tens()),
                  gridtools::cache<gridtools::K, gridtools::cache_io_policy::fill_and_flush, interval_start_0_end_0>(
                      p_u_tens())),
              gridtools::make_stage<stage_0_0>(p_u_tens(), p_v_tens(), p_u_nnow(), p_v_nnow(), p_fc())));
    }

    ~stencil_0() { m_stencil->finalize(); }

    gridtools::stencil* get_stencil() { return m_stencil.get(); }
  };

  // Members
  stencil_0 m_stencil_0;
  static constexpr const char* s_name = "coriolis_stencil";

 public:
  coriolis_stencil(const coriolis_stencil&) = delete;

  template <class S1, class S2, class S3, class S4, class S5>
  coriolis_stencil(const gridtools::clang::domain& dom, S1& u_tens, S2& u_nnow, S3& v_tens, S4& v_nnow, S5& fc)
      : m_stencil_0(dom, u_tens, u_nnow, v_tens, v_nnow, fc) {
    static_assert(gridtools::is_data_store<S1>::value,
                  "argument 'u_tens' is not a 'gridtools::data_store' (2nd argument invalid)");
    static_assert(gridtools::is_data_store<S2>::value,
                  "argument 'u_nnow' is not a 'gridtools::data_store' (3rd argument invalid)");
    static_assert(gridtools::is_data_store<S3>::value,
                  "argument 'v_tens' is not a 'gridtools::data_store' (4th argument invalid)");
    static_assert(gridtools::is_data_store<S4>::value,
                  "argument 'v_nnow' is not a 'gridtools::data_store' (5th argument invalid)");
    static_assert(gridtools::is_data_store<S5>::value,
                  "argument 'fc' is not a 'gridtools::data_store' (6th argument invalid)");
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
