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

namespace gridtools {

    namespace clang {

        struct temporary_storage;

        /**
         * @brief Dummy 3-dimensional storage
         * @ingroup gridtools_clang
         */
        struct storage {
            storage();

            template < typename T >
            storage(T...);

            storage &operator=(const storage &);
            storage &operator+=(const storage &);
            storage &operator-=(const storage &);
            storage &operator/=(const storage &);
            storage &operator*=(const storage &);

            storage &operator()(int, int, int);

            /**
             * @name 1D access
             * @{
             */
            storage &operator()(direction);
            storage &operator()(dimension);
            storage &operator()(offset);
            /** @} */

            /**
             * @name 2D access
             * @{
             */

            storage &operator()(dimension, dimension);

            storage &operator()(dimension, direction);
            storage &operator()(direction, dimension);
            storage &operator()(direction, direction);
            /** @} */

            /**
             * @name 3D access
             * @{
             */
            storage &operator()(dimension, dimension, dimension);

            storage &operator()(direction, direction, direction);
            storage &operator()(dimension, direction, direction);
            storage &operator()(direction, dimension, direction);
            storage &operator()(direction, direction, dimension);
            storage &operator()(dimension, dimension, direction);
            storage &operator()(dimension, direction, dimension);
            storage &operator()(direction, dimension, dimension);
            /** @} */

            operator double() const;
        };

        /**
         * @brief Dummy 3-dimensional temporary storage
         * @ingroup gridtools_clang
         */
        struct temporary_storage {
            temporary_storage();

            template < typename T >
            temporary_storage(T...);

            temporary_storage &operator=(const temporary_storage &);
            temporary_storage &operator+=(const temporary_storage &);
            temporary_storage &operator-=(const temporary_storage &);
            temporary_storage &operator/=(const temporary_storage &);
            temporary_storage &operator*=(const temporary_storage &);

            /**
             * @name 1D access
             * @{
             */
            temporary_storage &operator()(direction);
            temporary_storage &operator()(dimension);
            temporary_storage &operator()(offset);
            /** @} */

            /**
             * @name 2D access
             * @{
             */
            temporary_storage &operator()(dimension, dimension);

            temporary_storage &operator()(dimension, direction);
            temporary_storage &operator()(direction, dimension);
            temporary_storage &operator()(direction, direction);
            /** @} */

            /**
             * @name 3D access
             * @{
             */
            temporary_storage &operator()(dimension, dimension, dimension);

            temporary_storage &operator()(direction, direction, direction);
            temporary_storage &operator()(dimension, direction, direction);
            temporary_storage &operator()(direction, dimension, direction);
            temporary_storage &operator()(direction, direction, dimension);
            temporary_storage &operator()(dimension, dimension, direction);
            temporary_storage &operator()(dimension, direction, dimension);
            temporary_storage &operator()(direction, dimension, dimension);
            /** @} */

            operator double() const;
        };

#ifndef GRIDTOOLS_CLANG_META_DATA_T_DEFINED
        struct meta_data {
            template < typename T >
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
        using storage_ijk_t = storage;
        using storage_ij_t = storage;
        using storage_i_t = storage;
        using storage_j_t = storage;
        using storage_k_t = storage;
        using storage_scalar_t = storage;
#endif

#ifdef GRIDTOOLS_CLANG_GENERATED
        using ::gridtools::make_host_view;
#else
        template < typename T >
        storage_t make_host_view(T...);
#endif
    }
}
