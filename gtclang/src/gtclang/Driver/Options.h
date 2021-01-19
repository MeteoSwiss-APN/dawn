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
#include "dawn/CodeGen/Options.inc"
#include "dawn/Optimizer/Options.inc"
#include "dawn/Optimizer/PassOptions.inc"
#include "gtclang/Driver/Options.inc"
#undef OPT
};

/// @brief Configuration options used by gtclang and the DAWN library (most of them are parsed from
/// the command-line)
///
/// @ingroup driver
struct ParseOptions {
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  TYPE NAME = DEFAULT_VALUE;
  // clang-format off
OPT(bool, DumpPP, false, "dump-pp", "", "Dump the preprocessed code to stdout", "", false, false)
OPT(std::string, ConfigFile, "", "config", "",
    "json <file> with a single key \"globals\" which contains \"key\" : \"value\" pairs of global variables and their respective values. "
    "Global variables defined in the config file are treated as compile time constants and are replaced by their value",
    "<file>", true, false)
OPT(bool, DumpAST, false, "dump-ast", "", "Dump the clang AST of the preprocessed input to stdout", "", false, false)
OPT(bool, ReportPassPreprocessor, false, "report-pass-preprocessor", "",
    "Print each line of the preprocessed source prepended by the line number (comments and indentation are removed)", "", false, true)
  // clang-format on
#undef OPT
};

} // namespace gtclang

#endif
