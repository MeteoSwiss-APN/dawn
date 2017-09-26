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

#include "dawn/Optimizer/BoundaryExtent.h"
#include "dawn/Optimizer/DependencyGraphAccesses.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/Support/STLExtras.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dawn {

std::unique_ptr<std::unordered_map<std::size_t, Extents>>
computeBoundaryExtents(const DependencyGraphAccesses* graph) {
  using Vertex = DependencyGraphAccesses::Vertex;
  using Edge = DependencyGraphAccesses::Edge;

  const auto& adjacencyList = graph->getAdjacencyList();

  std::vector<std::size_t> nodesToVisit;
  std::unordered_set<std::size_t> visitedNodes;

  // Keep track of the extents of each vertex (and compute a VertexID to AccessID map)
  auto nodeExtentsPtr = make_unique<std::unordered_map<std::size_t, Extents>>();
  auto& nodeExtents = *nodeExtentsPtr;

  for(const auto& AccessIDVertexPair : graph->getVertices()) {
    const Vertex& vertex = AccessIDVertexPair.second;
    nodeExtents.emplace(vertex.VertexID, Extents{});
  }

  // Start from the output nodes and follow all paths
  for(std::size_t VertexID : graph->getOutputVertexIDs()) {
    nodesToVisit.clear();
    visitedNodes.clear();

    // Traverse all reachable nodes and update the node extents
    //
    // Consider the following example:
    //
    //              +-----+    <0, 1, 0, 1, 0, 0>     +-----+
    //              |  a  | ------------------------> |  b  |
    //              +-----+                           +-----+
    //      <-1, 1, 0, 0, 0, 0>                 <0, 0, -1, 0, 0, 0>
    //
    // If our current node is `a`, we compute the new extent of `b` by adding our current extent
    // to the edge extent (which represents the access pattern of b) and merge this result with
    // the already existing extent of `b`, thus:
    //
    // extent: b = merge(add(< -1, 1, 0, 0, 0, 0>, < 0, 1, 0, 1, 0, 0>), <0, 0, -1, 0, 0, 0>)
    //           = merge(<-1, 2, 0, 1, 0, 0>,  <0, 0, -1, 0, 0, 0>)
    //           = <-1, 2, -1, 1, 0, 0>
    //
    nodesToVisit.push_back(VertexID);
    while(!nodesToVisit.empty()) {

      // Process the current node
      std::size_t curNode = nodesToVisit.back();
      nodesToVisit.pop_back();
      const Extents& curExtent = nodeExtents[curNode];

      // Check if we already visited this node
      if(visitedNodes.count(curNode))
        continue;
      else
        visitedNodes.insert(curNode);

      // Follow edges of the current node and update the node extents
      for(const Edge& edge : *adjacencyList[curNode]) {
        nodeExtents[edge.ToVertexID].merge(Extents::add(curExtent, edge.Data));
        nodesToVisit.push_back(edge.ToVertexID);
      }
    }
  }

  return nodeExtentsPtr;
}

bool exceedsMaxBoundaryPoints(const DependencyGraphAccesses* graph,
                              int maxHorizontalBoundaryExtent) {
  auto nodeExtentsPtr = computeBoundaryExtents(graph);

  for(const auto& vertexIDExtentsPair : *nodeExtentsPtr) {
    const Extents& extents = vertexIDExtentsPair.second;
    if(extents[0].Plus > maxHorizontalBoundaryExtent ||
       extents[0].Minus < -maxHorizontalBoundaryExtent ||
       extents[1].Plus > maxHorizontalBoundaryExtent ||
       extents[1].Minus < -maxHorizontalBoundaryExtent)
      return true;
  }
  return false;
}

} // namespace dawn
