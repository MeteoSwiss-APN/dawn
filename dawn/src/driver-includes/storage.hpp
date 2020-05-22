//===--------------------------------------------------------------------------------*- C++ -*-===//
//                          _
//                         | |
//                       __| | __ ___      ___ ___
//                      / _` |/ _` \ \ /\ / / '_  |
//                     | (_| | (_| |\ V  V /| | | |
//                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#pragma once

#include "defs.hpp"
#include <type_traits>
#ifdef DAWN_GENERATED

// Set the storage type
#ifdef GRIDTOOLS_DAWN_HOST
#define DAWN_STORAGE_TYPE DAWN_STORAGE_HOST
#define BACKEND_MC 1
#endif

#ifdef GRIDTOOLS_DAWN_CUDA
#define DAWN_STORAGE_TYPE DAWN_STORAGE_CUDA
#define BACKEND_CUDA 1
#endif

// Default storage type is HOST
#ifndef DAWN_STORAGE_TYPE
#define DAWN_STORAGE_TYPE DAWN_STORAGE_HOST
#endif

// Default grid type is structured
#ifndef DAWN_GRID_TYPE
#define DAWN_GRID_TYPE DAWN_GRID_STRUCTURED
#endif

// Define grid
#if DAWN_GRID_TYPE == DAWN_GRID_STRUCTURED
#define STRUCTURED_GRIDS
#else
#error "DAWN_GRID_TYPE is invalid (only structured grids are supported)"
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

#if DAWN_STORAGE_TYPE == DAWN_STORAGE_CUDA
#define GT_USE_GPU
#endif

/**
 * @macro GRIDTOOLS_DAWN_PERFORMANCE_METERS
 * @name Compile the stencil with performance meters
 * @ingroup gridtools_dawn
 */
#ifdef GRIDTOOLS_DAWN_PERFORMANCE_METERS
#define ENABLE_METERS
#endif

// Include gridtools
#ifndef GRIDTOOLS_DAWN_NO_INCLUDE
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/storage/storage_facility.hpp>

#include "storage_runtime.hpp"
#endif

#ifdef GRIDTOOLS_DAWN_CUDA
#include "timer_cuda.hpp"
#endif

#endif
