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

#ifdef GRIDTOOLS_CLANG_DOXYGEN
/**
 * @namespace gridtools
 * @brief Namespace of the gridtools library
 */
namespace gridtools {
    /**
     * @namespace clang
     * @brief Namespace of the gridtools clang DSL
     */
    namespace clang {}
}

/**
 * @defgroup gridtools_clang gridtools_clang DSL
 * @brief gridtools clang DSL description
 */
#endif

#ifdef __clang__
#pragma clang system_header
#elif defined __GNUC__
#pragma GCC system_header
#endif

#include "gridtools/clang/boundary_condition.hpp"
#include "gridtools/clang/cxx11_warning.hpp"
#include "gridtools/clang/defs.hpp"
#include "gridtools/clang/domain.hpp"
#include "gridtools/clang/globals_impl.hpp"
#include "gridtools/clang/halo.hpp"
#include "gridtools/clang/interval.hpp"
#include "gridtools/clang/stencil.hpp"
#include "gridtools/clang/stencil_function.hpp"
#include "gridtools/clang/storage.hpp"
