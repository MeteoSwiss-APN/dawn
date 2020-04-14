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

#ifndef DAWN_OPTIMIZER_DRIVER_H
#define DAWN_OPTIMIZER_DRIVER_H

#include "dawn/Optimizer/Options.h"

#include <list>
#include <string>

namespace dawn {

/// @brief List of default optimizer pass groups
std::list<PassGroup> defaultPassGroups();

/// @brief Convert to/from string
/// {
PassGroup parsePassGroup(const std::string& passGroup);
std::string parsePassGroup(PassGroup passGroup);
/// }

/// TODO Driver methods will go here when OptimizerContext is removed.

} // namespace dawn

#endif
