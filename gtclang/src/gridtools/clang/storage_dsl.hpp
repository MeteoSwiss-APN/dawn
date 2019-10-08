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

#include "gridtools/clang/dimension.hpp"
#include "gridtools/clang/direction.hpp"
#include "gridtools/clang/interval.hpp"
#include "gridtools/clang/offset.hpp"
#include <type_traits>

#define STORAGE_CLASS_DEFN(Type)                                                                   \
  Type& operator=(const storage&);                                                                 \
  Type& operator+=(const storage&);                                                                \
  Type& operator-=(const storage&);                                                                \
  Type& operator/=(const storage&);                                                                \
  Type& operator*=(const storage&);                                                                \
                                                                                                   \
  /** @name 1D access @{ */                                                                        \
  Type& operator()(direction);                                                                     \
                                                                                                   \
  Type& operator()(dimension);                                                                     \
  Type& operator()(offset);                                                                        \
  /** @} */                                                                                        \
                                                                                                   \
  /** @name 2D access @{ */                                                                        \
  Type& operator()(dimension, dimension);                                                          \
                                                                                                   \
  Type& operator()(dimension, direction);                                                          \
  Type& operator()(direction, dimension);                                                          \
  Type& operator()(direction, direction);                                                          \
  /** @} */                                                                                        \
                                                                                                   \
  /** @name 3D access @{ */                                                                        \
  Type& operator()(dimension, dimension, dimension);                                               \
                                                                                                   \
  Type& operator()(direction, direction, direction);                                               \
  Type& operator()(dimension, direction, direction);                                               \
  Type& operator()(direction, dimension, direction);                                               \
  Type& operator()(direction, direction, dimension);                                               \
  Type& operator()(dimension, dimension, direction);                                               \
  Type& operator()(dimension, direction, dimension);                                               \
  Type& operator()(direction, dimension, dimension);                                               \
  /** @} */                                                                                        \
                                                                                                   \
  operator double() const;

namespace gridtools {

namespace clang {

/**
 * @brief Dummy 3-dimensional storage
 * @ingroup gridtools_clang
 */
struct storage {
  storage();

  template <typename T>
  storage(T...);

  storage& operator()(int, int, int);

  STORAGE_CLASS_DEFN(storage)
};

/**
 * @brief Dummy 1-dimensional i-storage
 * @ingroup gridtools_clang
 */
struct storage_i : public storage {
  storage_i();

  storage_i& operator()(int);

  STORAGE_CLASS_DEFN(storage_i);
};

/**
 * @brief Dummy 1-dimensional j-storage
 * @ingroup gridtools_clang
 */
struct storage_j : public storage {
  storage_j();

  storage_j& operator()(int);

  STORAGE_CLASS_DEFN(storage_j);
};

/**
 * @brief Dummy 1-dimensional k-storage
 * @ingroup gridtools_clang
 */
struct storage_k : public storage {
  storage_k();

  storage_k& operator()(int);

  STORAGE_CLASS_DEFN(storage_k);
};
/**
 * @brief Dummy 2-dimensional ij-storage
 * @ingroup gridtools_clang
 */
struct storage_ij : public storage {
  storage_ij();

  storage_ij& operator()(int, int);

  STORAGE_CLASS_DEFN(storage_ij);
};

/**
 * @brief Dummy 2-dimensional ik-storage
 * @ingroup gridtools_clang
 */
struct storage_ik : public storage {
  storage_ik();

  storage_ik& operator()(int, int);

  STORAGE_CLASS_DEFN(storage_ik);
};

/**
 * @brief Dummy 2-dimensional storage
 * @ingroup gridtools_clang
 */
struct storage_jk : public storage {
  storage_jk();

  storage_jk& operator()(int, int);

  STORAGE_CLASS_DEFN(storage_jk);
};

/**
 * @brief Dummy 3-dimensional storage
 * @ingroup gridtools_clang
 */
struct storage_ijk {
  storage_ijk();

  storage_ijk& operator()(int, int, int);

  STORAGE_CLASS_DEFN(storage_ijk);
};

struct var : public storage {
  var();

  var& operator()(int, int, int);

  STORAGE_CLASS_DEFN(var);
};

#undef STORAGE_CLASS_DEFN

#ifndef GRIDTOOLS_CLANG_META_DATA_T_DEFINED
struct meta_data {
  template <typename T>
  meta_data(T...);
};
using meta_data_t = meta_data;
using meta_data_ijk_t = meta_data;
using meta_data_ij_t = meta_data;
using meta_data_j_t = meta_data;
using meta_data_j_t = meta_data;
using meta_data_k_t = meta_data;
using meta_data_scalar_t = meta_data;
#endif

#ifndef GRIDTOOLS_CLANG_STORAGE_T_DEFINED
using storage_t = storage;
using storage_ijk_t = storage_ijk;
using storage_ij_t = storage_ij;
using storage_i_t = storage_i;
using storage_j_t = storage_j;
using storage_k_t = storage_k;
using storage_scalar_t = storage;
#endif

#ifdef GRIDTOOLS_CLANG_GENERATED
using ::gridtools::make_host_view;
#else
template <typename T>
storage_t make_host_view(T...);
#endif
} // namespace clang
} // namespace gridtools
