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
// clang-format off

// #define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP)
//
// TYPE           - Type of the option (bool, std::string or int)
// NAME           - Name of the option (must be a valid C++ identifier)
// DEFAULT_VALUE  - Default value if option was not passsed via command-line.
// OPTION         - Option for command-line (e.g OPTION == "XXX" generates a command-line target
//                  for --XXX and -XXX).
// OPTION_SHORT   - Option for POSIX style command-line. Set to empty NULL if unsued.
// HELP           - Help string.
// HELP_VALUE     - String of the value (e.g OPTION == "XXX" and HELP_VALUE == "<file>" will display
//                  the option as "--XXX=<file>"). Set to empty if unsued.
// HAS_VALUE      - Boolean if this option takes a value (only int and std::string options can take
//                  values).
// F_GROUP        - Boolean whether this option is part of the f-group. f-group options are
//                  prepended by an 'f' (e.g OPTION == "XXX" becomes "fXXX) an additional
//                  command-line target will be generated which allows to negate the option i.e
//

OPT(std::string, Backend, "gridtools", "backend", "",
    "Set backend code-generator. Possible values for <backend> are:"
    "\n - gridtools     = gridtools C++ code"
    "\n - c++-naive     = naive C++ code\n"
    "\n - c++-naive-ico = naive unstructured C++ code\n"
    "\n - c++-opt       = optimized C++ code"
    "\n - cuda          = optimized cuda", "<backend>", true, false)
OPT(std::string, OutputFile, "", "output", "o", "Write output to <file>", "<file>", true, false)
OPT(bool, SerializeIIR, false, "write-iir", "",
    "Serialize the low level intermediate representation after Optimization", "", false, false)
OPT(bool, DeserializeIIR, false, "read-iir", "",
    "Deserialize the low level intermediate representation from file", "", false, false)
OPT(std::string, IIRFormat, "json", "iir-format", "", "format of the output IIR", "", true, false)

// clang-format on
