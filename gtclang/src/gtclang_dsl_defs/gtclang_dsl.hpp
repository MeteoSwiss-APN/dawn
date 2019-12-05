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

#ifdef GTCLANG_DSL_DOXYGEN
/**
 * @namespace gtclang
 * @brief Namespace of gtclang frontend
 */
namespace gtclang {
/**
 * @namespace dsl
 * @brief Namespace of the gtclang DSL
 */
namespace dsl {}
} // namespace gtclang

/**
 * @defgroup gtclan_dsl gtclan_dsl DSL
 * @brief gtclang DSL description
 */
#endif

#ifdef __clang__
#pragma clang system_header
#elif defined __GNUC__
#pragma GCC system_header
#endif

#define BOOST_PP_VARIADICS 1

#include "gtclang_dsl_defs/boundary_condition.hpp"
#include "gtclang_dsl_defs/interval.hpp"
#include "gtclang_dsl_defs/stencil.hpp"
#include "gtclang_dsl_defs/stencil_function.hpp"
#include "gtclang_dsl_defs/storage_dsl.hpp"
