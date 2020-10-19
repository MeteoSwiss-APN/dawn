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

// clang-format off

// #define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP)
//
// TYPE           - Type of the option (bool, std::string or int)
// NAME           - Name of the option (must be a valid C++ identifier)
// DEFAULT_VALUE  - Default value if option was not passsed via command-line.
// OPTION         - Option for command-line (e.g OPTION == "XXX" generates a command-line target
//                  for --XXX and -XXX).
// OPTION_SHORT   - Option for POSIX style command-line. Set to empty if unsued.
// HELP           - Help string.
// HELP_VALUE     - String of the value (e.g OPTION == "XXX" and HELP_VALUE == "<file>" will display
//                  the option as "--XXX=<file>"). Set to empty if unsued.
// HAS_VALUE      - Boolean if this option takes a value (only int and std::string options can take
//                  values).
// F_GROUP        - Boolean whether this option is part of the f-group. f-group options are
//                  prepended by an 'f' (e.g OPTION == "XXX" becomes "fXXX) an additional
//                  command-line target will be generated which allows to negate the option i.e
//                  -fXXX and -fno-XXX.

OPT(bool, DumpSIR, false, "dump-sir", "", "Dump the parsed SIR to stdout", "", false, false)
OPT(bool, DumpAST, false, "dump-ast", "", "Dump the clang AST of the preprocessed input to stdout", "", false, false)
OPT(bool, WriteSIR, false, "write-sir", "", "Write the parsed SIR to stdout", "", false, false)
OPT(std::string, SIRFormat, "json", "sir-format", "", "format of the output SIR", "", true, false)
OPT(bool, DumpPP, false, "dump-pp", "", "Dump the preprocessed code to stdout", "", false, false)
OPT(std::string, ConfigFile, "", "config", "",
    "json <file> with a single key \"globals\" which contains \"key\" : \"value\" pairs of global variables and their respective values. "
    "Global variables defined in the config file are treated as compile time constants and are replaced by their value",
    "<file>", true, false)
OPT(bool, CodeGen, true, "codegen", "", "Generate code", "", false, true)
OPT(bool, ClangFormat, true, "clang-format", "", "Run clang-format on the generated code", "", false, true)
OPT(bool, ReportPassPreprocessor, false, "report-pass-preprocessor", "",
    "Print each line of the preprocessed source prepended by the line number (comments and indentation are removed)", "", false, true)
OPT(bool, Serialized, false, "loaded-serialized", "", "is the passed file serialized data", "", false, true)
OPT(bool, DisableOptimization, false, "no-opt", "",
    "Disables all optimisation", "", false, false)
OPT(bool, DefaultOptimization, false, "default-opt", "",
    "Runs all the default optimization", "", false, false)

OPT(std::string, Backend, "gridtools", "backend", "",
    "Set backend code-generator. Possible values for <backend> are:"
    "\n - gridtools     = gridtools C++ code"
    "\n - c++-naive     = naive C++ code\n"
    "\n - c++-naive-ico = naive unstructured C++ code\n"
    "\n - c++-opt       = optimized C++ code"
    "\n - cuda          = optimized cuda", "<backend>", true, false)
OPT(std::string, OutputFile, "", "output", "o", "Write output to <file>", "<file>", true, false)

// clang-format on
