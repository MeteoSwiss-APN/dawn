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

#ifndef DAWN_OPTIMIZER_OPTIMIZEROPTIONS_H
#define DAWN_OPTIMIZER_OPTIMIZEROPTIONS_H

namespace dawn {

// TODO Rename this to dawn::Options after the larger dawn::Options is removed
struct OptimizerOptions {
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  TYPE NAME = DEFAULT_VALUE;
#include "dawn/Optimizer/Options.inc"
#undef OPT
};

} // namespace dawn

#endif
