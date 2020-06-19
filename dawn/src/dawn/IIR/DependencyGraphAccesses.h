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

#include "dawn/IIR/ASTFwd.h"
#include "dawn/IIR/DependencyGraph.h"
#include "dawn/IIR/Extents.h"
#include "dawn/Support/TypeTraits.h"
#include <functional>
#include <set>
#include <unordered_map>

namespace dawn {
class OptimizerContext;

namespace iir {
class Accesses;
class StencilMetaInformation;

/// @enum DependencyGraphAccessesEdgeData
/// @brief Edges contain the extent of the access between the two nodes
/// @ingroup optimizer
using DependencyGraphAccessesEdgeData = Extents;

/// @brief Dependency graph of the accesses
///
/// Field, variable and literal accesses are represented as nodes and connected via edges which
/// contain the access extent. Consider the following example:
///
/// @code{.cpp}
///   a(i, j, k) = 5.0;
///   b(i, k, k) = a(i+1, j-1, k);
/// @endcode
///
/// which results in the following dependency graph
///
/// @dot
/// digraph example {
///   node [ shape=record, fontname=Helvetica, fontsize=10 ];
///   a -> 5.0 [ style="dashed" ];
///   b -> a [ label="  <0, 1, -1, 0, 0, 0>" ];
/// }
/// @enddot
///
/// @ingroup optimizer
class DependencyGraphAccesses
    : public DependencyGraph<DependencyGraphAccesses, DependencyGraphAccessesEdgeData> {

  std::reference_wrapper<const StencilMetaInformation> metaData_;
  std::unordered_map<std::size_t, int> VertexIDToAccessIDMap_;

public:
  using Base = DependencyGraph<DependencyGraphAccesses, DependencyGraphAccessesEdgeData>;
  using EdgeData = DependencyGraphAccessesEdgeData;

  DependencyGraphAccesses(const StencilMetaInformation& metaData) : Base(), metaData_(metaData) {}

  /// @brief Construct graph by merging the given `graphs`
  ///
  /// @param graphs       Graphs to merge
  /// @tparam GraphTypes  Variadic pack of `DependencyGraphAccesses`
  template <class... GraphTypes>
  DependencyGraphAccesses(const StencilMetaInformation& metaData, const GraphTypes&... graphs)
      : DependencyGraphAccesses(metaData) {
    static_assert(and_<std::is_same<GraphTypes, DependencyGraphAccesses>...>::value,
                  "GraphTypes needs to be a variadic pack of `DependencyGraphAccesses`");
    for(const auto& g : {graphs...})
      merge(g);
  }

  bool operator==(const DependencyGraphAccesses& other) const {
    return Base::operator==(other) && (VertexIDToAccessIDMap_ == other.VertexIDToAccessIDMap_);
  }

  /// @brief Process the statement and insert it into the current graph
  ///
  /// For each write and read access a node will be inserted. Between each write and read access
  /// an edge will be created s.t
  ///
  /// +-------+           +--------+
  /// | Write | --------> |  Read  |
  /// +-------+           +--------+
  ///
  /// Note that only child-less nodes are processed.
  void insertStatement(const std::shared_ptr<iir::Stmt>& stmt);

  /// @brief Insert a new node
  Vertex& insertNode(int ID);

  /// @brief Merge extents if edge already exists
  void edgeAlreadyExists(EdgeData& existingEdge, const EdgeData& newEdge);

  /// @brief Get the ID of the vertex given by ID
  int getIDFromVertexID(std::size_t VertexID) const;

  /// @brief EdgeData to string
  const char* edgeDataToString(const EdgeData& data) const;

  /// @brief EdgeData to dot
  std::string edgeDataToDot(const EdgeData& data) const;

  /// @brief Get the shape of the dot grahps
  const char* getDotShape() const;

  /// @brief Get the name of the vertex given by ID
  std::string getVertexNameByVertexID(std::size_t VertexID) const;

  /// @brief Merge `other` into `this`
  void merge(const DependencyGraphAccesses& other);

  /// @brief Partition the graph into the non-connected sub-graphs
  std::vector<std::set<std::size_t>> partitionInSubGraphs() const;

  /// @brief Check if the graph is a DAG
  ///
  /// If the graph consists of non-connected subgraphs, it will check that each subgraph is a DAG.
  /// In our context, a DAG is defined as having a non-empty set of input as well as output nodes.
  bool isDAG() const;

  /// @brief Get the VertexIDs of the pure `output` vertices
  ///
  /// Output vertices are vertices which do not have incoming edges from other vertices.
  std::vector<std::size_t> getOutputVertexIDs() const;

  /// @brief Get the VertexIDs of the pure `input` vertices
  ///
  /// Input vertices are vertices which only have outgoing edges i.e no incoming edges. This will
  /// almost always contain all the literal accesses.
  std::vector<std::size_t> getInputVertexIDs() const;

  /// @brief Find @b multi-node strongly connected components (SCC) in the accesses graph using
  /// Tarjan's algorithm
  ///
  /// @param scc      The output set of multi-node strongly connected components (given as a vector
  ///                 of vectors of AccessIDs). The vector needs to be allocated but will be cleared
  ///                 upon invocation.
  /// @returns `true` if atleast one multi-node SCC was found, `false` otherwise.
  ///
  /// @see https://en.wikipedia.org/wiki/Tarjan's_strongly_connected_components_algorithm
  bool findStronglyConnectedComponents(std::vector<std::set<int>>& scc) const;

  /// @brief Check if the accesses graph contains any @b multi-node strongly connected components
  /// using Tarjan's algorithm
  ///
  /// @returns `true` if atleast one multi-node SCC was found, `false` otherwise.
  ///
  /// @see https://en.wikipedia.org/wiki/Tarjan's_strongly_connected_components_algorithm
  bool hasStronglyConnectedComponents() const;

  /// @brief Greedily colors the graph
  ///
  /// @param coloring   Map of AccessID to assigned color
  ///
  /// @see https://en.wikipedia.org/wiki/Greedy_coloring
  void greedyColoring(std::unordered_map<int, int>& coloring) const;

  /// @brief Clear the graph
  void clear();

  /// @brief Serialize the graph to JSON
  void toJSON(const std::string& file) const;

  /// @fn exceedsMaxBoundaryPoints
  /// @brief Check if any field, referenced in `graph`, exceeds the maximum number of boundary
  /// points in the @b horizontal
  /// @ingroup optimizer
  bool exceedsMaxBoundaryPoints(int maxHorizontalBoundaryExtent);
};
} // namespace iir
} // namespace dawn
