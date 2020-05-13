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

// TODO Rename this to (MaxHaloPoints, "max-halo") after combined struct is eliminated
OPT(int, MaxHaloSize, 3, "max-halo-size", "",
    "Set the maximum number of allowed halo points", "<N>", true, false)
OPT(bool, UseParallelEP, false, "use-parallel-ep", "",
    "Make use of the parallel execution policy", "", false, true)
OPT(bool, RunWithSync, true, "run-with-sync", "",
    "Generate run methods with storage synchronization", "", false, true)
OPT(int, MaxBlocksPerSM, 0, "max-blocks-sm", "",
    "Maximum number of blocks that can be registered per SM", "<max-blocks-sm>", true, false)
OPT(int, nsms, 0, "nsms", "", "Number of (CUDA) SMs", "<nsms>", true, false)
OPT(int, DomainSizeI, 0, "domain-size-i", "", "i domain size for compiler optimization", "", true, false)
OPT(int, DomainSizeJ, 0, "domain-size-j", "", "j domain size for compiler optimization", "", true, false)
OPT(int, DomainSizeK, 0, "domain-size-k", "", "k domain size for compiler optimization", "", true, false)

// clang-format on
