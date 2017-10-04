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

#ifndef GTCLANG_FRONTEND_OPTIONS_H
#define GTCLANG_FRONTEND_OPTIONS_H

#include <string>

namespace gtclang {

/// @brief Configuration options used by gtclang and the DAWN library (most of them are parsed from
/// the command-line)
///
/// @ingroup driver
struct Options {
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  TYPE NAME = DEFAULT_VALUE;
#include "dawn/Compiler/Options.inc"
#include "gtclang/Driver/Options.inc"
#undef OPT
};

} // namespace gtclang

#endif
