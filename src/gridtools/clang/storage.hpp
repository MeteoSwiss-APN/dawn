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

#include <type_traits>
#ifdef GRIDTOOLS_CLANG_GENERATED

// Set the storage type
#ifdef GRIDTOOLS_CLANG_HOST
#define GRIDTOOLS_CLANG_STORAGE_TYPE GRIDTOOLS_CLANG_STORAGE_HOST
#endif

#ifdef GRIDTOOLS_CLANG_CUDA
#define GRIDTOOLS_CLANG_STORAGE_TYPE GRIDTOOLS_CLANG_STORAGE_CUDA
#endif

// Default storage type is HOST
#ifndef GRIDTOOLS_CLANG_STORAGE_TYPE
#define GRIDTOOLS_CLANG_STORAGE_TYPE GRIDTOOLS_CLANG_STORAGE_HOST
#endif

// Default grid type is structured
#ifndef GRIDTOOLS_CLANG_GRID_TYPE
#define GRIDTOOLS_CLANG_GRID_TYPE GRIDTOOLS_CLANG_GRID_STRUCTURED
#endif

// Define grid
#if GRIDTOOLS_CLANG_GRID_TYPE == GRIDTOOLS_CLANG_GRID_STRUCTURED
#define STRUCTURED_GRIDS
#else
#error "GRIDTOOLS_CLANG_GRID_TYPE is invalid (only structured grids are supported)"
#endif

// gridtools specific typedefs
#ifndef CXX11_ENABLED
#define CXX11_ENABLED
#endif

#ifndef SUPPRESS_MESSAGES
#define SUPPRESS_MESSAGES
#endif

#ifndef PEDANTIC_DISABLED
#define PEDANTIC_DISABLED
#endif

#if GRIDTOOLS_CLANG_STORAGE_TYPE == GRIDTOOLS_CLANG_STORAGE_CUDA
#define _USE_GPU_
#endif

/**
 * @macro GRIDTOOLS_CLANG_PERFORMANCE_METERS
 * @name Compile the stencil with performance meters
 * @ingroup gridtools_clang
 */
#ifdef GRIDTOOLS_CLANG_PERFORMANCE_METERS
#define ENABLE_METERS
#endif

// Include gridtools
#ifndef GRIDTOOLS_CLANG_NO_INCLUDE
#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/stencil-composition/stencil-functions/call_interfaces.hpp>
#include <gridtools/storage/storage-facility.hpp>
#endif

#include "gridtools/clang/storage_runtime.hpp"

#ifdef GRIDTOOLS_CLANG_CUDA
#include "gridtools/clang/timer_cuda.hpp"
#endif

#endif

#include "gridtools/clang/storage_dsl.hpp"
