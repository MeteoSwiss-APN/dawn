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

#ifndef DAWN_OPTIMIZER_STRONGLYCONNECTEDCOMPONENTS_H
#define DAWN_OPTIMIZER_STRONGLYCONNECTEDCOMPONENTS_H

#include "dawn/Optimizer/DependencyGraphAccessesFwd.h"
#include <set>
#include <stack>
#include <unordered_map>
#include <vector>

namespace dawn {

/// @brief Find strongly connected components using Tarjan's algorithm
///
/// @see https://en.wikipedia.org/wiki/Tarjan's_strongly_connected_components_algorithm
class StronglyConnectedComponents {

  /// @brief Auxiliary data stored for each vertex
  struct VertexData {
    /// Unique index of each vertex, which numbers the nodes consecutively in the order in which
    /// they are discovered
    int Index;

    /// The smallest index of any node known to be reachable from the vertex, including the vertex
    /// itself. Therefore the vertex must be left on the stack if `Lowlink < Index`, whereas the
    /// vertex must be removed as the root of a strongly connected component if `Lowlink == Index`.
    /// The value `Lowlink` is computed during the depth-first search from the vertex, as this
    /// finds the nodes that are reachable from this vertex.
    int LowLink;

    /// Check if the node is currently on the stack
    bool OnStack;

    /// Reference to the vertex
    int ID;
  };

  std::unordered_map<std::size_t, VertexData> vertexData_;
  std::stack<std::size_t> vertexStack_;

  int index_;
  bool hasMultiNodeSCC_;

  const DependencyGraphAccesses* graph_;
  std::vector<std::set<int>>* scc_;

public:
  StronglyConnectedComponents(const DependencyGraphAccesses* graph, std::vector<std::set<int>>* scc)
      : index_(0), hasMultiNodeSCC_(false), graph_(graph), scc_(scc) {}

  /// @brief Find the strongly connected components
  bool find();

private:
  void findImpl(std::size_t FromVertexID);
};
} // namespace dawn

#endif
