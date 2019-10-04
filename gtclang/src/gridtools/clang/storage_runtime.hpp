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

#pragma once

#ifndef GRIDTOOLS_CLANG_GENERATED
#error "This file should only be used in generated code!"
#endif

#define GRIDTOOLS_CLANG_META_DATA_T_DEFINED
#define GRIDTOOLS_CLANG_STORAGE_T_DEFINED

#include <gridtools/storage/storage_facility.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/stencil_composition/stencil_functions.hpp>

namespace gridtools {

namespace clang {

/**
 * @name Runtime storage environment
 * @ingroup gridtools_clang
 * @{
 */

#ifdef GRIDTOOLS_CLANG_HALO_EXTEND
using halo_ijk_t = gridtools::halo<halo::value, halo::value, 0>;
using halo_ij_t = gridtools::halo<halo::value, halo::value, 0>;
using halo_i_t = gridtools::halo<halo::value, 0, 0>;
using halo_j_t = gridtools::halo<0, halo::value, 0>;
#else
using halo_ijk_t = gridtools::halo<0, 0, 0>;
using halo_ij_t = gridtools::halo<0, 0, 0>;
using halo_i_t = gridtools::halo<0, 0, 0>;
using halo_j_t = gridtools::halo<0, 0, 0>;
#endif

/**
  * @brief Backend type
  */
#ifdef __CUDACC__
using backend_t = gridtools::backend::cuda;
#else
using backend_t = gridtools::backend::mc;
#endif

using storage_traits_t = gridtools::storage_traits<backend_t>;
/**
 * @brief Meta-data types
 * @{
 */
using meta_data_ijk_t = storage_traits_t::storage_info_t<0, 3, halo_ij_t>;
using meta_data_ij_t =
    storage_traits_t::special_storage_info_t<1, gridtools::selector<1, 1, 0>, halo_ij_t>;
using meta_data_i_t =
    storage_traits_t::special_storage_info_t<2, gridtools::selector<1, 0, 0>, halo_i_t>;
using meta_data_j_t =
    storage_traits_t::special_storage_info_t<3, gridtools::selector<0, 1, 0>, halo_j_t>;
using meta_data_k_t = storage_traits_t::special_storage_info_t<4, gridtools::selector<0, 0, 1>>;
using meta_data_scalar_t =
    storage_traits_t::special_storage_info_t<5, gridtools::selector<0, 0, 0>>;
using meta_data_t = meta_data_ijk_t;
/** @} */

/**
 * @brief Storage types
 * @{
 */
using storage_ijk_t = storage_traits_t::data_store_t<float_type, meta_data_ijk_t>;
using storage_ij_t = storage_traits_t::data_store_t<float_type, meta_data_ij_t>;
using storage_i_t = storage_traits_t::data_store_t<float_type, meta_data_i_t>;
using storage_j_t = storage_traits_t::data_store_t<float_type, meta_data_j_t>;
using storage_k_t = storage_traits_t::data_store_t<float_type, meta_data_k_t>;
using storage_scalar_t = storage_traits_t::data_store_t<float_type, meta_data_scalar_t>;
using storage_t = storage_ijk_t;
/** @} */

/** @} */

#if GRIDTOOLS_CLANG_STORAGE_TYPE == GRIDTOOLS_CLANG_STORAGE_HOST
#define GT_BACKEND_DECISION_viewmaker(x) make_host_view(x)
#define GT_BACKEND_DECISION_bcapply gridtools::boundary_apply
#elif GRIDTOOLS_CLANG_STORAGE_TYPE == GRIDTOOLS_CLANG_STORAGE_CUDA
#define GT_BACKEND_DECISION_viewmaker(x) make_device_view(x)
#define GT_BACKEND_DECISION_bcapply gridtools::boundary_apply_gpu
#endif
}
}
