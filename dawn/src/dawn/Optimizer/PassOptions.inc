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

// Boolean parameters that turn on and off passes and specific pass reports. These will be removed.

OPT(bool, DefaultNone, false, "default-none", "",
    "Selectively run passes", "", false, true)
OPT(bool, Parallel, false, "parallel", "",
    "Run parallel pass group", "", false, true)
OPT(bool, SSA, false, "ssa", "",
    "Transform all statements into static single assignment (SSA) form", "", false, false)
OPT(bool, PrintStencilGraph, false, "print-stencil-graph", "",
    "Dump the access dependency graph of each stencil to a dot file", "", false, true)
OPT(bool, SetStageName, false, "set-stage-name", "",
    "Run print-stage-name pass group", "", false, true)
OPT(bool, StageReordering, false, "stage-reordering", "",
    "Run reorder-stages pass group", "", false, true)
OPT(bool, StageMerger, false, "stage-merger", "",
    "Merge stages within a multi-stage into the same Do-Method if possible", "", false, true)
OPT(bool, TemporaryMerger, false, "temporary-merger", "",
    "Merge temporaries if possible", "", false, true)
OPT(bool, Inlining, false, "inlining", "",
    "Run inlining pass group", "", false, true)
OPT(bool, IntervalPartitioning, false, "interval-partitioning", "",
    "partitions the intervals so there are no overlapping doMethods anymore", "", false, true)
OPT(bool, TmpToStencilFunction, false, "tmp-to-stencil-function", "",
    "Activate pass to replace temporary precomputations by stencil function calls", "", false, true)
OPT(bool, SetNonTempCaches, false, "set-non-temp-caches", "",
    "Run set-non-temp pass group", "", false, true)
OPT(bool, SetCaches, false, "set-caches", "",
    "Run set-caches pass group", "", false, true)
OPT(bool, SetBlockSize, false, "set-block-size", "",
    "Run set-block-size pass group", "", false, true)
OPT(bool, SetLoopOrder, false, "opt-loop-order", "",
    "Optimizes loop order to be parallel if possible", "", false, true)    
OPT(bool, DataLocalityMetric, false, "data-locality-metric", "",
    "Run data-locality-metric pass group", "", false, true)

OPT(bool, ReportBoundaryConditions, false, "report-bc", "",
    "Report where boundary conditions are inserted", "", false, true)
OPT(bool, ReportDataLocalityMetric, false, "report-dl", "",
    "Compute and report the data-locality metric for each stencil", "", false, true)
OPT(bool, ReportPassTmpToFunction, false, "report-pass-tmp-to-function", "",
    "Detailed report on the actions taken during the replace temporary by stencil function call pass", "", false, true)
OPT(bool, ReportPassRemoveScalars, false, "report-pass-remove-scalars", "",
    "Report which variables are removed by the scalar removal pass", "", false, true)
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
OPT(bool, ReportPassSetBlockSize, false, "report-pass-set-block-size", "",
    "Report on the pass that sets the block size", "", false, true)
OPT(bool, ReportPassSetNonTempCaches, false, "report-cache-non-temp-fields", "",
    "Report which non temporary fields are cached for each multi-stage", "", false, true)

// clang-format on
