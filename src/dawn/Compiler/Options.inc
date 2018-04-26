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
OPT(std::string, InlineStrategy, "cof", "inline", "", 
    "Set the strategy used to inline stencil functions. Possible values for <strategy> are:"
    "\n - none = Disable inlining"
    "\n - cof  = Favor computations on the fly"
    "\n - pc   = Favor pre-computations\n", "<strategy>", true, false)
OPT(std::string, ReorderStrategy, "greedy", "reorder", "", 
    "Set the strategy used to reorder the stages (or statements) of the stencils. Possible values for <strategy> are:"
    "\n - none   = Disable reordering"
    "\n - greedy = Use greedy fusing"
    "\n - scut   = Use S-cut graph partitioning\n", "<strategy>", true, false)
OPT(int, MaxFieldsPerStencil, 40, "max-fields", "",
    "Set the maximum number of fields in any given stencils", "<N>", true, false)
OPT(bool, SSA, false, "ssa", "",
    "Transform all statements into static single assigment (SSA) form", "", false, true)
OPT(bool, DumpSplitGraphs, false, "dump-split-dags", "", 
    "Dump the access dependency graph of all stages and multi-stages during the splitting passes to dot files", "", false, true)
OPT(bool, DumpStencilGraph, false, "dump-stencil-dag", "", 
    "Dump the initial access dependency graph of each stencil to a dot file", "", false, true)
OPT(bool, DumpStageGraph, false, "dump-stage-dag", "", 
    "Dump the initial dependency graph of the stages of each stencil to a dot file", "", false, true)
OPT(bool, DumpTemporaryGraphs, false, "dump-tmp-dag", "", 
    "Dump the dependency graph of the temporaries of each stencil to a dot file", "", false, true)
OPT(bool, DumpRaceConditionGraph, false, "dump-rc-dag", "", 
    "In case an unresolvable race-condition is detected, dump the dependency graph to a dot file", "", false, true)
OPT(bool, KeepVarnames, false, "keep-varnames", "", 
    "Keep the names of locally defined variables (this should merely be used for debugging as it may result in invalid code)", "", false, true)
OPT(bool, ReportDataLocalityMetric, false, "report-dl", "", 
    "Compute and report the data-locality metric for each stencil", "", false, true)
OPT(bool, MergeTemporaries, false, "merge-temporaries", "", 
    "Merge temporaries if possible", "", false, true)
OPT(bool, SplitStencils, false, "split-stencils", "", 
    "Split stencil whose number of fields exceeds a threshold", "", false, true)
OPT(bool, MergeStages, false, "merge-stages", "", 
    "Merge stages within a multi-stage into the same Do-Method if possible", "", false, true)
OPT(bool, MergeDoMethods, true, "merge-do-methods", "", 
    "Merge Do-Methods with different vertical intervals into the same stage if possible", "", false, true) 
OPT(bool, UseParallelEP, false, "use-parallel-ep", "", 
    "Make use of the parallel execution policy", "", false, true)
OPT(bool, UseKCaches, true, "use-kcaches", "", 
    "Make use of the k-caches", "", false, true)
OPT(bool, PassTmpToFunction, false, "pass-tmp-to-function", "",
    "Activate pass to replace temporary precomputations by stencil function calls", "", false, true)
OPT(bool, ReportPassTmpToFunction, false, "report-pass-tmp-to-function", "",
    "Detailed report on the actions taken during the replace temporary by stencil function call pass", "", false, true)
OPT(bool, ReportAccesses, false, "report-accesses", "", 
    "Detailed report on the accesses of each statement", "", false, true)
OPT(bool, ReportPassStageSplit, false, "report-pass-stage-split", "", 
    "Detailed report on the actions taken during the stage splitting pass", "", false, true)
OPT(bool, ReportPassMultiStageSplit, false, "report-pass-multi-stage-split", "", 
    "Detailed report on the actions taken during the multi-stage splitting pass", "", false, true)
OPT(bool, ReportPassFieldVersioning, false, "report-pass-field-versioning", "", 
    "Report on all field renamings during the field-versioning pass", "", false, true)
OPT(bool, ReportPassTemporaryMerger, false, "report-pass-temporary-merger", "", 
    "Report which temporaries will be merged during the temporary merger pass. The groups of temporaries, "
    "which will be merged, are sorted in alphapetical order.", "", false, true)
OPT(bool, ReportPassTemporaryType, false, "report-pass-temporary-type", "", 
    "Report which variables and temporary fields are promoted/demoted", "", false, true)    
OPT(bool, ReportPassStageReodering, false, "report-pass-stage-reordering", "", 
    "Dump the stencil-instantiation before and after the stage reordering pass to json", "", false, true)
OPT(bool, ReportPassStageMerger, false, "report-pass-stage-merger", "", 
    "Dump the stencil-instantiation before and after the stage merger pass to JSON", "", false, true)
OPT(bool, ReportPassSetCaches, false, "report-pass-set-caches", "", 
    "Report which fields are cached for each multi-stage", "", false, true)
OPT(bool, UseNonTempCaches, false, "cache-non-temp-fields", "",
    "Allows for caching of non-temporary fields", "", false, true)
OPT(bool, ReportPassSetNonTempCaches, false, "report-cache-non-temp-fields", "",
    "Report which non temporary fields are cached for each multi-stage", "", false, true)
OPT(bool, ReportBoundaryConditions, false, "report-bc", "",
    "Report where boundary conditions are inserted", "", false, true)
OPT(bool, Debug, false, "debug", "",
    "Compile to debug backend", "", false, true)
// clang-format on
