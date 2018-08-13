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

#ifndef DAWN_OPTIMIZER_ACCESSUTILS_H
#define DAWN_OPTIMIZER_ACCESSUTILS_H

#include "dawn/IIR/Accesses.h"
#include "dawn/IIR/Field.h"
#include <unordered_map>

namespace dawn {
namespace AccessUtils {

/// @brief given a write access, with AccessID, it will recorded in the corresponding map of input,
/// output or inputOutput
/// depending on previous accesses to the same field
///
/// @ingroup optimizer
void recordWriteAccess(std::unordered_map<int, Field>& inputOutputFields,
                       std::unordered_map<int, Field>& inputFields,
                       std::unordered_map<int, Field>& outputFields, int AccessID,
                       const boost::optional<Extents>& extents, Interval const& doMethodInterval);

/// @brief given a read access, with AccessID, it will recorded in the corresponding map of input,
/// output or inputOutput
/// depending on previous accesses to the same field
///
/// @ingroup optimizer
void recordReadAccess(std::unordered_map<int, Field>& inputOutputFields,
                      std::unordered_map<int, Field>& inputFields,
                      std::unordered_map<int, Field>& outputFields, int AccessID,
                      const boost::optional<Extents>& extents, Interval const& doMethodInterval);

} // namespace AccessUtils
} // namespace dawn

#endif
