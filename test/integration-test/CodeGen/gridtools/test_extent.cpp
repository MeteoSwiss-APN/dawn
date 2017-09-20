//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _     _ _              _            _
//                        (_)   | | |            | |          | |
//               __ _ _ __ _  __| | |_ ___   ___ | |___    ___| | __ _ _ __   __ _
//              / _` | '__| |/ _` | __/ _ \ / _ \| / __|  / __| |/ _` | '_ \ / _` |
//             | (_| | |  | | (_| | || (_) | (_) | \__ \ | (__| | (_| | | | | (_| |
//              \__, |_|  |_|\__,_|\__\___/ \___/|_|___/  \___|_|\__,_|_| |_|\__, |
//               __/ |                                                        __/ |
//              |___/                                                        |___/
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

// RUN: %c++% %file% %gridtools_flags% -fsyntax-only

// This file can be used to test the gridtools extent analysis. Change the placeholder you are 
// intrested in:
//
//    using final_extent = typename boost::mpl::at<final_map, p_pp_in>::type;
//
// (here it is `p_pp_in`) and compilation will fail and print the extent.
// 
// This file can be run with gtclang-tester (just pass this directory as an argument to the bash 
// scripts in the gridtools_clang build directory).

#define GRIDTOOLS_CLANG_GENERATED 1
#define GRIDTOOLS_CLANG_HALO_EXTEND 3
#ifndef BOOST_RESULT_OF_USE_TR1
#define BOOST_RESULT_OF_USE_TR1 1
#endif
#ifndef BOOST_NO_CXX11_DECLTYPE
#define BOOST_NO_CXX11_DECLTYPE 1
#endif
#ifndef BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS 1
#endif
#ifndef FUSION_MAX_VECTOR_SIZE
#define FUSION_MAX_VECTOR_SIZE 30
#endif
#ifndef FUSION_MAX_MAP_SIZE
#define FUSION_MAX_MAP_SIZE 30
#endif
#ifndef BOOST_MPL_LIMIT_VECTOR_SIZE
#define BOOST_MPL_LIMIT_VECTOR_SIZE 30
#endif

#include "gridtools/clang_dsl.hpp"

using namespace gridtools;
using namespace enumtype;

typedef interval<level<0, -1>, level<1, -1>> x_interval;
typedef interval<level<0, -1>, level<1, -1>> interval_start_0_end_0;

struct stage_0_0 {
  using __tmp_lap_57 = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
  using pp_in = gridtools::accessor<1, gridtools::enumtype::in, gridtools::extent<-1, 1, -1, 1, 0, 0>>;
  using crlato = gridtools::accessor<2, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
  using crlatu = gridtools::accessor<3, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
  using arg_list = boost::mpl::vector<__tmp_lap_57, pp_in, crlato, crlatu>;

  template <typename Evaluation>
  GT_FUNCTION static void Do(Evaluation const& eval, interval_start_0_end_0) {}
};

struct stage_0_1 {
  using pp_out = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
  using __tmp_lap_42 = gridtools::accessor<1, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
  using w_in = gridtools::accessor<2, gridtools::enumtype::in, gridtools::extent<-1, 1, -1, 1, 0, 0>>;
  using pp_in = gridtools::accessor<3, gridtools::enumtype::in, gridtools::extent<-1, 1, -1, 1, 0, 0>>;
  using crlato = gridtools::accessor<4, gridtools::enumtype::in, gridtools::extent<0, 0, -1, 0, 0, 0>>;
  using crlatu = gridtools::accessor<5, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
  using hdmask = gridtools::accessor<6, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
  using __tmp_lap_57 = gridtools::accessor<7, gridtools::enumtype::in, gridtools::extent<-1, 1, -1, 1, 0, 0>>;
  using arg_list = boost::mpl::vector<pp_out, __tmp_lap_42, w_in, pp_in, crlato, crlatu, hdmask, __tmp_lap_57>;

  template <typename Evaluation>
  GT_FUNCTION static void Do(Evaluation const& eval, interval_start_0_end_0) {}
};

struct stage_0_2 {
  using w_out = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
  using __tmp_lap_27 = gridtools::accessor<1, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
  using v_in = gridtools::accessor<2, gridtools::enumtype::in, gridtools::extent<-1, 1, -1, 1, 0, 0>>;
  using w_in = gridtools::accessor<3, gridtools::enumtype::in, gridtools::extent<-1, 1, -1, 1, 0, 0>>;
  using crlato = gridtools::accessor<4, gridtools::enumtype::in, gridtools::extent<0, 0, -1, 0, 0, 0>>;
  using crlatu = gridtools::accessor<5, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
  using hdmask = gridtools::accessor<6, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
  using __tmp_lap_42 = gridtools::accessor<7, gridtools::enumtype::in, gridtools::extent<-1, 1, -1, 1, 0, 0>>;
  using arg_list = boost::mpl::vector<w_out, __tmp_lap_27, v_in, w_in, crlato, crlatu, hdmask, __tmp_lap_42>;

  template <typename Evaluation>
  GT_FUNCTION static void Do(Evaluation const& eval, interval_start_0_end_0) {}
};

struct stage_0_3 {
  using v_out = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
  using __tmp_lap_12 = gridtools::accessor<1, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
  using u_in = gridtools::accessor<2, gridtools::enumtype::in, gridtools::extent<-1, 1, -1, 1, 0, 0>>;
  using v_in = gridtools::accessor<3, gridtools::enumtype::in, gridtools::extent<-1, 1, -1, 1, 0, 0>>;
  using crlato = gridtools::accessor<4, gridtools::enumtype::in, gridtools::extent<0, 0, -1, 0, 0, 0>>;
  using crlatu = gridtools::accessor<5, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
  using hdmask = gridtools::accessor<6, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
  using __tmp_lap_27 = gridtools::accessor<7, gridtools::enumtype::in, gridtools::extent<-1, 1, -1, 1, 0, 0>>;
  using arg_list = boost::mpl::vector<v_out, __tmp_lap_12, u_in, v_in, crlato, crlatu, hdmask, __tmp_lap_27>;

  template <typename Evaluation>
  GT_FUNCTION static void Do(Evaluation const& eval, interval_start_0_end_0) {}
};

struct stage_0_4 {
  using u_out = gridtools::accessor<0, gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, 0, 0>>;
  using u_in = gridtools::accessor<1, gridtools::enumtype::in, gridtools::extent<-1, 1, -1, 1, 0, 0>>;
  using crlato = gridtools::accessor<2, gridtools::enumtype::in, gridtools::extent<0, 0, -1, 0, 0, 0>>;
  using hdmask = gridtools::accessor<3, gridtools::enumtype::in, gridtools::extent<0, 0, 0, 0, 0, 0>>;
  using __tmp_lap_12 = gridtools::accessor<4, gridtools::enumtype::in, gridtools::extent<-1, 1, -1, 1, 0, 0>>;
  using arg_list = boost::mpl::vector<u_out, u_in, crlato, hdmask, __tmp_lap_12>;

  template <typename Evaluation>
  GT_FUNCTION static void Do(Evaluation const& eval, interval_start_0_end_0) {}
};

#define BACKEND backend<Host, GRIDBACKEND, Block>

typedef layout_map<2, 1, 0> layout_t;
typedef BACKEND::storage_info<0, layout_t> storage_info_type;

typedef BACKEND::storage_type<float_type, storage_info_type>::type storage_type;
typedef BACKEND::temporary_storage_type<float_type, storage_info_type>::type temporary_storage_type;

// Domain
using p___tmp_lap_57 = gridtools::arg<0, temporary_storage_type>;
using p___tmp_lap_42 = gridtools::arg<1, temporary_storage_type>;
using p___tmp_lap_27 = gridtools::arg<2, temporary_storage_type>;
using p___tmp_lap_12 = gridtools::arg<3, temporary_storage_type>;
using p_u_out = gridtools::arg<4, storage_type>;
using p_v_out = gridtools::arg<5, storage_type>;
using p_w_out = gridtools::arg<6, storage_type>;
using p_pp_out = gridtools::arg<7, storage_type>;
using p_u_in = gridtools::arg<8, storage_type>;
using p_v_in = gridtools::arg<9, storage_type>;
using p_w_in = gridtools::arg<10, storage_type>;
using p_pp_in = gridtools::arg<11, storage_type>;
using p_crlato = gridtools::arg<12, storage_type>;
using p_crlatu = gridtools::arg<13, storage_type>;
using p_hdmask = gridtools::arg<14, storage_type>;

using domain_arg_list =
    boost::mpl::vector<p___tmp_lap_57, p___tmp_lap_42, p___tmp_lap_27, p___tmp_lap_12, p_u_out, p_v_out, p_w_out,
                       p_pp_out, p_u_in, p_v_in, p_w_in, p_pp_in, p_crlato, p_crlatu, p_hdmask>;

int main() {
  using functor1 = decltype(gridtools::make_stage<stage_0_0>(p___tmp_lap_57(), p_pp_in(), p_crlato(), p_crlatu()));
  using functor2 = decltype(gridtools::make_stage<stage_0_1>(p_pp_out(), p___tmp_lap_42(), p_w_in(), p_pp_in(), p_crlato(),
                                                             p_crlatu(), p_hdmask(), p___tmp_lap_57()));
  using functor3 = decltype(make_stage<stage_0_2>(p_w_out(), p___tmp_lap_27(), p_v_in(), p_w_in(), p_crlato(), p_crlatu(),
                                                  p_hdmask(), p___tmp_lap_42()));
  using functor4 = decltype(make_stage<stage_0_3>(p_v_out(), p___tmp_lap_12(), p_u_in(), p_v_in(), p_crlato(), p_crlatu(),
                                                  p_hdmask(), p___tmp_lap_27()));
  using functor5 = decltype(make_stage<stage_0_4>(p_u_out(), p_u_in(), p_crlato(), p_hdmask(), p___tmp_lap_12()));
  using mss_t = decltype(
      gridtools::make_multistage(
          enumtype::execute<enumtype::forward>(),
          functor1(),
          functor2(),
          functor3(),
          functor4(),
          functor5()));

  using placeholders = boost::mpl::vector<p___tmp_lap_57, p___tmp_lap_42, p___tmp_lap_27, p___tmp_lap_12, p_u_out, p_v_out, p_w_out,
                                          p_pp_out, p_u_in, p_v_in, p_w_in, p_pp_in, p_crlato, p_crlatu, p_hdmask>;
  typedef compute_extents_of<init_map_of_extents<placeholders>::type, 1>::for_mss<mss_t>::type final_map;

  using final_extent = typename boost::mpl::at<final_map, p_pp_in>::type;
  final_extent::trigger_error tmp;

  return 0;
}

