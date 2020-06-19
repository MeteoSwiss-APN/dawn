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

namespace dawn {
namespace codegen {

/// @brief CodeGen backends
enum class Backend { GridTools, CXXNaive, CXXNaiveIco, CUDAIco, CUDA, CXXOpt };

/// @brief Options for all codegen backends combined.
///
/// These have to be disjoint from the options for other dawn and gtc-parse components.
struct Options {
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  TYPE NAME = DEFAULT_VALUE;
#include "dawn/CodeGen/Options.inc"
#undef OPT
};

} // namespace codegen
} // namespace dawn
