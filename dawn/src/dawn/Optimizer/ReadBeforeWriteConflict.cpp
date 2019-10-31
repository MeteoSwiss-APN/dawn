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

#include "dawn/Optimizer/ReadBeforeWriteConflict.h"
#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/IIR/Extents.h"
#include "dawn/Support/Assert.h"
#include <unordered_set>
#include <utility>
#include <vector>

namespace dawn {

namespace {

/// @brief Detect read-before-write conflicts
///
/// The algorithm assumes a well formed stencil graph meaning the graph contains at *least* one
/// output field. An output field is given by a node on which no other node depends on, except
/// possibly itself.
template <bool IsVertical>
class ReadBeforeWriteConflictDetector {
  const iir::DependencyGraphAccesses* graph_;
  iir::LoopOrderKind loopOrder_;

public:
  ReadBeforeWriteConflictDetector(const iir::DependencyGraphAccesses* graph,
                                  iir::LoopOrderKind loopOrder)
      : graph_(graph), loopOrder_(loopOrder) {}

  ReadBeforeWriteConflict check() const {

    std::vector<std::size_t> nodesToVisit = graph_->getOutputVertexIDs();
    DAWN_ASSERT_MSG(!nodesToVisit.empty(), "invalid graph (probably contains cycles!)");

    ReadBeforeWriteConflict conflict;
    for(std::size_t VertexID : nodesToVisit) {
      conflict |= checkVertex(VertexID);

      // If we have detected a conflict in the loop and counter-loop order we can stop
      if(conflict.LoopOrderConflict && conflict.CounterLoopOrderConflict)
        return conflict;
    }

    return conflict;
  }

private:
  ReadBeforeWriteConflict checkVertex(std::size_t VertexID) const {
    ReadBeforeWriteConflict conflict;

    std::vector<std::size_t> nodesToVisit;
    std::unordered_set<std::size_t> visitedNodes;
    const auto& adjacencyList = graph_->getAdjacencyList();

    nodesToVisit.push_back(VertexID);
    while(!nodesToVisit.empty()) {

      // Process next node
      std::size_t curNode = nodesToVisit.back();
      nodesToVisit.pop_back();

      // Check if we already visited this node
      if(visitedNodes.count(curNode))
        continue;
      else
        visitedNodes.insert(curNode);

      // Follow edges of the current node
      if(!adjacencyList[curNode]->empty()) {
        for(const auto& edge : *adjacencyList[curNode]) {
          const iir::Extents& extent = edge.Data;

          if(IsVertical) {

            if(!adjacencyList[edge.ToVertexID]->empty()) {

              // We have an outgoing edge to a non-input field, check the vertical accesses
              auto verticalLoopOrderAccess = extent.getVerticalLoopOrderAccesses(loopOrder_);

              if(verticalLoopOrderAccess.CounterLoopOrder)
                // We have a conflict in the counter loop-order
                conflict.CounterLoopOrderConflict = true;
              else if(verticalLoopOrderAccess.LoopOrder)
                // We have a conflict in the loop-order
                conflict.LoopOrderConflict = true;
            }

            // If we have detected a conflict in the loop and counter-loop order we can stop
            if(conflict.LoopOrderConflict && conflict.CounterLoopOrderConflict)
              return conflict;

          } else {

            // Check if we have a non-pointwise (i.e stencil) outgoing edge ...
            if(!extent.isHorizontalPointwise()) {

              // ... to a non-input field (i.e an intermediate field or variable)
              if(!adjacencyList[edge.ToVertexID]->empty()) {
                // We have a read-after-write conflict -> exit
                return ReadBeforeWriteConflict(true, true);
              }
            }
          }

          // Continue visiting the nodes
          nodesToVisit.push_back(edge.ToVertexID);
        }
      }
    }

    return conflict;
  }
};

} // anonymous namespace

ReadBeforeWriteConflict::ReadBeforeWriteConflict()
    : LoopOrderConflict(false), CounterLoopOrderConflict(false) {}

ReadBeforeWriteConflict::ReadBeforeWriteConflict(bool loopOrderConflict,
                                                 bool counterLoopOrderConflict)
    : LoopOrderConflict(loopOrderConflict), CounterLoopOrderConflict(counterLoopOrderConflict) {}

ReadBeforeWriteConflict& ReadBeforeWriteConflict::operator|=(const ReadBeforeWriteConflict& other) {
  LoopOrderConflict |= other.LoopOrderConflict;
  CounterLoopOrderConflict |= other.CounterLoopOrderConflict;
  return *this;
}

ReadBeforeWriteConflict
hasVerticalReadBeforeWriteConflict(const iir::DependencyGraphAccesses* graph,
                                   iir::LoopOrderKind loopOrder) {
  return ReadBeforeWriteConflictDetector<true>(graph, loopOrder).check();
}

bool hasHorizontalReadBeforeWriteConflict(const iir::DependencyGraphAccesses* graph) {
  return ReadBeforeWriteConflictDetector<false>(graph, iir::LoopOrderKind::Parallel /* unused */)
      .check()
      .LoopOrderConflict;
}

} // namespace dawn
