// gtclang (0.0.1-a685a2b-x86_64-linux-gnu-7.3.0)
// based on LLVM/Clang (3.8.1), Dawn (0.0.1)
// Generated on 2018-10-27  19:24:20

#define GRIDTOOLS_CLANG_GENERATED 1
#define GRIDTOOLS_CLANG_BACKEND_T GT
#ifndef BOOST_RESULT_OF_USE_TR1
 #define BOOST_RESULT_OF_USE_TR1 1
#endif
#ifndef BOOST_NO_CXX11_DECLTYPE
 #define BOOST_NO_CXX11_DECLTYPE 1
#endif
#ifndef GRIDTOOLS_CLANG_HALO_EXTEND
 #define GRIDTOOLS_CLANG_HALO_EXTEND 3
#endif
#ifndef BOOST_PP_VARIADICS
 #define BOOST_PP_VARIADICS 1
#endif
#ifndef BOOST_FUSION_DONT_USE_PREPROCESSED_FILES
 #define BOOST_FUSION_DONT_USE_PREPROCESSED_FILES 1
#endif
#ifndef BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
 #define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS 1
#endif
#ifndef GT_VECTOR_LIMIT_SIZE
 #define GT_VECTOR_LIMIT_SIZE 40
#endif
#ifndef BOOST_FUSION_INVOKE_MAX_ARITY
 #define BOOST_FUSION_INVOKE_MAX_ARITY GT_VECTOR_LIMIT_SIZE
#endif
#ifndef FUSION_MAX_VECTOR_SIZE
 #define FUSION_MAX_VECTOR_SIZE GT_VECTOR_LIMIT_SIZE
#endif
#ifndef FUSION_MAX_MAP_SIZE
 #define FUSION_MAX_MAP_SIZE GT_VECTOR_LIMIT_SIZE
#endif
#ifndef BOOST_MPL_LIMIT_VECTOR_SIZE
 #define BOOST_MPL_LIMIT_VECTOR_SIZE GT_VECTOR_LIMIT_SIZE
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

class compute_extent_test_stencil {
 public:
  struct stencil_7 {
    // Intervals
    using interval_start_0_end_0 = gridtools::interval<gridtools::level<0, 1>, gridtools::level<1, -1>>;
    using axis_stencil_7 = gridtools::interval<gridtools::level<0, -1>, gridtools::level<1, 1>>;
    using grid_stencil_7 = gridtools::grid<axis_stencil_7>;

    struct stage_0_0 {
      using u = gridtools::accessor<0, gridtools::enumtype::in, gridtools::extent<-1, 1, -1, 1, 0, 0>>;
      using lap = gridtools::accessor<1, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using arg_list = boost::mpl::vector<u, lap>;

      template <typename Evaluation>
      GT_FUNCTION static void Do(Evaluation& eval, interval_start_0_end_0) {
        eval(lap(0, 0, 0)) = ((((eval(u(1, 0, 0)) + eval(u(-1, 0, 0))) + eval(u(0, 1, 0))) + eval(u(0, -1, 0))) -
                              ((gridtools::clang::float_type)4 * eval(u(0, 0, 0))));
      }
    };

    struct stage_0_1 {
      using lap = gridtools::accessor<0, gridtools::enumtype::in, gridtools::extent<0, 1, 0, 0, 0, 0>>;
      using u = gridtools::accessor<1, gridtools::enumtype::in, gridtools::extent<0, 1, 0, 0, 0, 0>>;
      using flx = gridtools::accessor<2, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using fly = gridtools::accessor<3, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using arg_list = boost::mpl::vector<lap, u, flx, fly>;

      template <typename Evaluation>
      GT_FUNCTION static void Do(Evaluation& eval, interval_start_0_end_0) {
        if (((eval(flx(0, 0, 0)) * (eval(u(1, 0, 0)) - eval(u(0, 0, 0)))) > (int)0)) {
          eval(flx(0, 0, 0)) = (gridtools::clang::float_type)0;
        } else {
          eval(flx(0, 0, 0)) = (eval(lap(1, 0, 0)) - eval(lap(0, 0, 0)));
        }
        if (((eval(fly(0, 0, 0)) * (eval(u(0, 1, 0)) - eval(u(0, 0, 0)))) > (int)0)) {
          eval(fly(0, 0, 0)) = (gridtools::clang::float_type)0;
        } else {
          eval(fly(0, 0, 0)) = (eval(lap(0, 1, 0)) - eval(lap(0, 0, 0)));
        }
      }
    };

    struct stage_0_2 {
      using flx = gridtools::accessor<0, gridtools::enumtype::in, gridtools::extent<-1, 0, 0, 0, 0, 0>>;
      using fly = gridtools::accessor<1, gridtools::enumtype::in, gridtools::extent<0, 0, -1, 0, 0, 0>>;
      using u = gridtools::accessor<2, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using coeff = gridtools::accessor<3, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using out = gridtools::accessor<4, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
      using arg_list = boost::mpl::vector<flx, fly, u, coeff, out>;

      template <typename Evaluation>
      GT_FUNCTION static void Do(Evaluation& eval, interval_start_0_end_0) {
        eval(out(0, 0, 0)) =
            (eval(u(0, 0, 0)) -
             (eval(coeff(0, 0, 0)) *
              (((eval(flx(0, 0, 0)) - eval(flx(-1, 0, 0))) + eval(fly(0, 0, 0))) - eval(fly(0, -1, 0)))));
      }
    };

    stencil_7(const gridtools::clang::domain& dom, storage_ijk_t coeff, storage_ijk_t out, storage_ijk_t u) {
      // Check if extents do not exceed the halos
      static_assert((static_cast<int>(storage_ijk_t::storage_info_t::halo_t::template at<0>()) >= 0) ||
                        (storage_ijk_t::storage_info_t::layout_t::template at<0>() == -1),
                    "Used extents exceed halo limits.");
      static_assert(((-1) * static_cast<int>(storage_ijk_t::storage_info_t::halo_t::template at<0>()) <= 0) ||
                        (storage_ijk_t::storage_info_t::layout_t::template at<0>() == -1),
                    "Used extents exceed halo limits.");
      static_assert((static_cast<int>(storage_ijk_t::storage_info_t::halo_t::template at<1>()) >= 0) ||
                        (storage_ijk_t::storage_info_t::layout_t::template at<1>() == -1),
                    "Used extents exceed halo limits.");
      static_assert(((-1) * static_cast<int>(storage_ijk_t::storage_info_t::halo_t::template at<1>()) <= 0) ||
                        (storage_ijk_t::storage_info_t::layout_t::template at<1>() == -1),
                    "Used extents exceed halo limits.");
      static_assert((static_cast<int>(storage_ijk_t::storage_info_t::halo_t::template at<0>()) >= 0) ||
                        (storage_ijk_t::storage_info_t::layout_t::template at<0>() == -1),
                    "Used extents exceed halo limits.");
      static_assert(((-1) * static_cast<int>(storage_ijk_t::storage_info_t::halo_t::template at<0>()) <= 0) ||
                        (storage_ijk_t::storage_info_t::layout_t::template at<0>() == -1),
                    "Used extents exceed halo limits.");
      static_assert((static_cast<int>(storage_ijk_t::storage_info_t::halo_t::template at<1>()) >= 0) ||
                        (storage_ijk_t::storage_info_t::layout_t::template at<1>() == -1),
                    "Used extents exceed halo limits.");
      static_assert(((-1) * static_cast<int>(storage_ijk_t::storage_info_t::halo_t::template at<1>()) <= 0) ||
                        (storage_ijk_t::storage_info_t::layout_t::template at<1>() == -1),
                    "Used extents exceed halo limits.");
      static_assert((static_cast<int>(storage_ijk_t::storage_info_t::halo_t::template at<0>()) >= 2) ||
                        (storage_ijk_t::storage_info_t::layout_t::template at<0>() == -1),
                    "Used extents exceed halo limits.");
      static_assert(((-1) * static_cast<int>(storage_ijk_t::storage_info_t::halo_t::template at<0>()) <= -2) ||
                        (storage_ijk_t::storage_info_t::layout_t::template at<0>() == -1),
                    "Used extents exceed halo limits.");
      static_assert((static_cast<int>(storage_ijk_t::storage_info_t::halo_t::template at<1>()) >= 1) ||
                        (storage_ijk_t::storage_info_t::layout_t::template at<1>() == -1),
                    "Used extents exceed halo limits.");
      static_assert(((-1) * static_cast<int>(storage_ijk_t::storage_info_t::halo_t::template at<1>()) <= -2) ||
                        (storage_ijk_t::storage_info_t::layout_t::template at<1>() == -1),
                    "Used extents exceed halo limits.");
      using p_coeff = gridtools::arg<0, storage_ijk_t>;
      using p_out = gridtools::arg<1, storage_ijk_t>;
      using p_fly = gridtools::tmp_arg<2, storage_t>;
      using p_flx = gridtools::tmp_arg<3, storage_t>;
      using p_lap = gridtools::tmp_arg<4, storage_t>;
      using p_u = gridtools::arg<5, storage_ijk_t>;
      using domain_arg_list = boost::mpl::vector<p_coeff, p_out, p_fly, p_flx, p_lap, p_u>;

      // Grid
      gridtools::halo_descriptor di = {dom.iminus(), dom.iminus(), dom.iplus(), dom.isize() - 1 - dom.iplus(),
                                       dom.isize()};
      gridtools::halo_descriptor dj = {dom.jminus(), dom.jminus(), dom.jplus(), dom.jsize() - 1 - dom.jplus(),
                                       dom.jsize()};
      auto grid_ = grid_stencil_7(di, dj);
      grid_.value_list[0] = dom.kminus();
      grid_.value_list[1] = dom.ksize() == 0 ? 0 : dom.ksize() - dom.kplus();
      coeff.sync();
      out.sync();
      u.sync();

      // Computation
      m_stencil = gridtools::make_computation<gridtools::clang::backend_t>(
          grid_, (p_coeff() = coeff), (p_out() = out), (p_u() = u),
          gridtools::make_multistage(
              gridtools::enumtype::execute<gridtools::enumtype::forward /*parallel*/>(),
              gridtools::define_caches(gridtools::cache<gridtools::IJ, gridtools::cache_io_policy::local>(p_fly()),
                                       gridtools::cache<gridtools::IJ, gridtools::cache_io_policy::local>(p_lap()),
                                       gridtools::cache<gridtools::IJ, gridtools::cache_io_policy::local>(p_flx())),
              gridtools::make_stage_with_extent<stage_0_0, extent<-1, 1, -1, 0>>(p_u(), p_lap()),
              gridtools::make_stage_with_extent<stage_0_1, extent<-1, 0, -1, 0>>(p_lap(), p_u(), p_flx(), p_fly()),
              gridtools::make_stage_with_extent<stage_0_2, extent<0, 0, 0, 0>>(p_flx(), p_fly(), p_u(), p_coeff(),
                                                                               p_out())));
    }

    // Members
    computation<void> m_stencil;

    computation<void>* get_stencil() { return &m_stencil; }
  };

  // Stencil-Data
  const gridtools::clang::domain& m_dom;
  static constexpr const char* s_name = "compute_extent_test_stencil";

  // Members representing all the stencils that are called
  stencil_7 m_stencil_7;

 public:
  compute_extent_test_stencil(const compute_extent_test_stencil&) = delete;

  compute_extent_test_stencil(const gridtools::clang::domain& dom, storage_ijk_t u, storage_ijk_t out,
                              storage_ijk_t coeff)
      : m_dom(dom), m_stencil_7(dom, coeff, out, u) {}

  void run() { m_stencil_7.get_stencil()->run(); }

  std::string get_name() const { return std::string(s_name); }

  std::vector<computation<void>*> getStencils() {
    return std::vector<gridtools::computation<void>*>({m_stencil_7.get_stencil()});
  }

  void reset_meters() { m_stencil_7.get_stencil()->reset_meter(); }
};
}  // namespace gridtool
;
