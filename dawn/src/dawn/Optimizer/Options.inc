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

OPT(int, MaxHaloPoints, 3, "max-halo", "",
    "Set the maximum number of allowed halo points", "<N>", true, false)
OPT(std::string, ReorderStrategy, "greedy", "reorder", "",
    "Set the strategy used to reorder the stages (or statements) of the stencils. Possible values for <strategy> are:\n - none   = Disable reordering\n - greedy = Use greedy fusing\n - scut   = Use S-cut graph partitioning\n",
    "<strategy>", true, false)
OPT(int, MaxFieldsPerStencil, 40, "max-fields", "",
    "Set the maximum number of fields in any given stencils", "<N>", true, false)
OPT(bool, MaxCutMSS, false, "max-cut-mss", "",
    "Cuts the given multistages in as many multistages as possible while maintaining legal code", "", false, true)
OPT(int, BlockSizeI, 0, "block-size-i", "", "i block size for tiled computations", "", true, false)
OPT(int, BlockSizeJ, 0, "block-size-j", "", "j block size for tiled computations", "", true, false)
OPT(int, BlockSizeK, 0, "block-size-k", "", "k block size for tiled computations", "", true, false)

// Hardware options
OPT(int, SMemMaxFields, 8, "smem-max-fields", "",
    "Maximum number of fields concurrently in shared memory", "", true, false)
OPT(int, TexCacheMaxFields, 3, "tex-cache-max-fields", "",
    "Maximum number of fields concurrently in the texture cache", "", true, false)

OPT(bool, SplitStencils, false, "split-stencils", "",
    "Split stencil whose number of fields exceeds a threshold", "", false, true)
OPT(bool, MergeStages, false, "merge-stages", "",
    "Merge stages within a multi-stage into the same Do-Method if possible", "", false, true)
OPT(bool, MergeDoMethods, true, "merge-do-methods", "",
    "Merge Do-Methods with different vertical intervals into the same stage if possible", "", false, true)
OPT(bool, DisableKCaches, false, "disable-kcaches", "",
    "Disable use of the k-caches", "", false, true)
OPT(bool, KeepVarnames, false, "keep-varnames", "",
    "Keep the names of locally defined variables (this should merely be used for debugging as it may result in invalid code)", "", false, true)

OPT(bool, ReportAccesses, false, "report-accesses", "",
    "Detailed report on the accesses of each statement", "", false, true)

OPT(bool, SerializeIIR, false, "write-iir", "",
    "Serialize the low level intermediate representation after Optimization", "", false, false)
OPT(std::string, IIRFormat, "json", "iir-format", "", "format of the output IIR", "", true, false)

OPT(bool, DumpSplitGraphs, false, "dump-split-dags", "",
    "Dump the access dependency graph of all stages and multi-stages during the splitting passes to dot files", "", false, true)
OPT(bool, DumpStageGraph, false, "dump-stage-dag", "",
    "Dump the initial dependency graph of the stages of each stencil to a dot file", "", false, true)
OPT(bool, DumpTemporaryGraphs, false, "dump-tmp-dag", "",
    "Dump the dependency graph of the temporaries of each stencil to a dot file", "", false, true)
OPT(bool, DumpRaceConditionGraph, false, "dump-rc-dag", "",
    "In case an unresolvable race-condition is detected, dump the dependency graph to a dot file", "", false, true)
OPT(bool, DumpStencilInstantiation, false, "dump-si", "",
    "Dump every stencil instantiation to the standard output", "", false, true)
OPT(bool, WriteStencilInstantiation, false, "write-si", "",
    "Write stencil instantiation to JSON files before and after stage reordering", "", false, true)
OPT(bool, DumpStencilGraph, false, "dump-stencil-dag", "",
    "Dump the initial access dependency graph of each stencil to a dot file", "", false, true)

// clang-format on
