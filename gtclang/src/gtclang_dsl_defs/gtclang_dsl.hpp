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
} // namespace gridtools

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

#define BOOST_PP_VARIADICS 1

#include "gtclang_dsl_defs/boundary_condition.hpp"
#include "gtclang_dsl_defs/globals_impl.hpp"
#include "gtclang_dsl_defs/interval.hpp"
#include "gtclang_dsl_defs/stencil.hpp"
#include "gtclang_dsl_defs/stencil_function.hpp"
#include "gtclang_dsl_defs/storage_dsl.hpp"
