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

#ifndef DAWN_OPTIMIZER_BOUNDARYEXTENTS_H
#define DAWN_OPTIMIZER_BOUNDARYEXTENTS_H

#include "dawn/Optimizer/Extents.h"
#include "dawn/Optimizer/Field.h"
#include <memory>
#include <unordered_map>

namespace dawn {

namespace iir {
class DependencyGraphAccesses;
}

/// @fn computeBoundaryPoints
/// @brief Compute the accumulated extent of each Vertex (given by `VertexID`) referenced in `graph`
/// @returns map of `VertexID` to boundary extent
/// @ingroup optimizer
extern std::unique_ptr<std::unordered_map<std::size_t, Extents>>
computeBoundaryExtents(const iir::DependencyGraphAccesses* graph);

/// @fn exceedsMaxBoundaryPoints
/// @brief Check if any field, referenced in `graph`, exceeds the maximum number of boundary points
/// in the @b horizontal
/// @ingroup optimizer
extern bool exceedsMaxBoundaryPoints(const iir::DependencyGraphAccesses* graph,
                                     int maxHorizontalBoundaryExtent);

} // namespace dawn

#endif
