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

#pragma once

#include "dawn/IIR/LoopOrder.h"

namespace dawn {

namespace iir {
class DependencyGraphAccesses;
}

/// @brief Result of the vertical dependency analysis algorithm
/// @ingroup optimizer
struct ReadBeforeWriteConflict {
  bool LoopOrderConflict;        ///< Conflict in the loop-order
  bool CounterLoopOrderConflict; ///< Conflict in the counter loop-order

  ReadBeforeWriteConflict();
  ReadBeforeWriteConflict(bool loopOrderConflict, bool counterLoopOrderConflict);
  ReadBeforeWriteConflict& operator|=(const ReadBeforeWriteConflict& other);
};

/// @brief Check if the graph contains any vertical non-pointwise read-before-write conflicts in
/// the loop- or counter-loop-order.
///
/// The algorithm will check if there are two nodes which are connected by a non-pointwise edge
/// with the first node being an output field (i.e no other nodes depend on it) and the second
/// node being @b not an input field (i.e an other node depends on it). This simply means we
/// detect non-pointwise read-before-writes. Consider the following example:
///
/// @code
///   vertical_region(k_start, k_end - 1}) {
///     lap = u;
///     out = lap(k+1);
///   }
/// @endcode
///
/// As we have a forward loop, we read `lap` at `k+1` before it has been written! To bypass this
/// problem, we have to split the two statements into two different k-loops (i.e multi-stages).
///
/// @code
///   vertical_region(k_start, k_end - 1)
///     lap = u;
///
///   vertical_region(k_start, k_end - 1)
///     out = lap(k+1);
/// @endcode
///
/// Note that in this scenario the first k-loop could be executed in parallel as there are no
/// vertical dependencies.
///
/// @note
/// If there are no conflicts the graph defines a valid MultiStage.
///
/// @see MultiStage
///
/// @ingroup optimizer
ReadBeforeWriteConflict
hasVerticalReadBeforeWriteConflict(const iir::DependencyGraphAccesses& graph,
                                   iir::LoopOrderKind loopOrder);

/// @brief Check if the graph contains any horizontal non-pointwise read-before-write conflicts
///
/// The algorithm will check if there are two nodes which are connected by a non-pointwise edge
/// with the first node being an output field (i.e no other nodes depend on it) and the second
/// node being @b not an input field (i.e other node depends on it). This simply means we
/// detect non-pointwise read-before-writes. Consider the following example:
///
/// @code
///   lap = u;
///   out = lap(i+1);
/// @endcode
///
/// We will potentially read `lap` at `i+1` before it has been written! This is due to the fact
/// that our parallel model assumes each statement is embarrassingly parallel in the horizontal.
/// The solution is is thus to insert a synchronization point between the two statements.
///
/// @code
///   lap = u;
///   __syncthreads():
///   out = lap(i+1);
/// @endcode
///
/// This can be formally achieved by splitting the two statements into a different stages.
///
/// @note
/// If there are no conflicts the graph defines a valid Stage.
///
/// @see Stage
///
/// @ingroup optimizer
bool hasHorizontalReadBeforeWriteConflict(const iir::DependencyGraphAccesses& graph);

} // namespace dawn
