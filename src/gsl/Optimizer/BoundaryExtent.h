//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#ifndef GSL_OPTIMIZER_BOUNDARYEXTENTS_H
#define GSL_OPTIMIZER_BOUNDARYEXTENTS_H

#include "gsl/Optimizer/Extents.h"
#include "gsl/Optimizer/Field.h"
#include <memory>
#include <unordered_map>

namespace gsl {

class DependencyGraphAccesses;

/// @fn computeBoundaryPoints
/// @brief Compute the accumulated extent of each Vertex (given by `VertexID`) referenced in `graph`
/// @returns map of `VertexID` to boundary extent
/// @ingroup optimizer
extern std::unique_ptr<std::unordered_map<std::size_t, Extents>>
computeBoundaryExtents(const DependencyGraphAccesses* graph);

/// @fn exceedsMaxBoundaryPoints
/// @brief Check if any field, referenced in `graph`, exceeds the maximum number of boundary points
/// in the @b horizontal
/// @ingroup optimizer
extern bool exceedsMaxBoundaryPoints(const DependencyGraphAccesses* graph,
                                     int maxHorizontalBoundaryExtent);

} // namespace gsl

#endif
