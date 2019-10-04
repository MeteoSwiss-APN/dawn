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

#ifndef DAWN_IIR_ACCESSUTILS_H
#define DAWN_IIR_ACCESSUTILS_H

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
void recordWriteAccess(std::unordered_map<int, iir::Field>& inputOutputFields,
                       std::unordered_map<int, iir::Field>& inputFields,
                       std::unordered_map<int, iir::Field>& outputFields, int AccessID,
                       const std::optional<iir::Extents>& extents,
                       iir::Interval const& doMethodInterval);

/// @brief given a read access, with AccessID, it will recorded in the corresponding map of input,
/// output or inputOutput
/// depending on previous accesses to the same field
///
/// @ingroup optimizer
void recordReadAccess(std::unordered_map<int, iir::Field>& inputOutputFields,
                      std::unordered_map<int, iir::Field>& inputFields,
                      std::unordered_map<int, iir::Field>& outputFields, int AccessID,
                      const std::optional<iir::Extents>& extents,
                      iir::Interval const& doMethodInterval);

} // namespace AccessUtils
} // namespace dawn

#endif // DAWN_IIR_ACCESSUTILS_H
