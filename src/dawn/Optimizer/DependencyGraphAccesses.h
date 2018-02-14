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

#ifndef DAWN_OPTIMIZER_DEPENDENCYGRAPHACCESSES_H
#define DAWN_OPTIMIZER_DEPENDENCYGRAPHACCESSES_H

#include "dawn/Compiler/DiagnosticsMessage.h"
#include "dawn/Optimizer/BoundaryExtent.h"
#include "dawn/Optimizer/DependencyGraph.h"
#include "dawn/Optimizer/Extents.h"
#include "dawn/Optimizer/GreedyColoring.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/StatementAccessesPair.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/Optimizer/StronglyConnectedComponents.h"
#include "dawn/Support/Json.h"
#include "dawn/Support/StringUtil.h"
#include "dawn/Support/TypeTraits.h"
#include <set>
#include <unordered_map>

namespace dawn {

class Accesses;
class OptimizerContext;
class StencilInstantiation;
class StatementAccessesPair;
class Stmt;

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
template <typename VertexData>
class DependencyGraphAccessesT
    : public DependencyGraph<DependencyGraphAccessesT<VertexData>, DependencyGraphAccessesEdgeData,
                             VertexData> {

  StencilInstantiation* instantiation_;
  std::unordered_map<std::size_t, int> VertexIDToAccessIDMap_;

  template <typename IntT, typename VertexDataT>
  struct vertex_has_data {
    static constexpr bool value = (std::is_void<VertexDataT>::value &&
                                   std::is_integral<typename std::decay<IntT>::type>::value);
  };

public:
  using Base = DependencyGraph<DependencyGraphAccessesT<VertexData>,
                               DependencyGraphAccessesEdgeData, VertexData>;

  using EdgeData = DependencyGraphAccessesEdgeData;
  using Vertex = typename Base::Vertex;
  using VertexDataEmul = typename Base::VertexDataEmul;
  using Edge = typename Base::Edge;
  using EdgeList = typename Base::EdgeList;

  using Base::vertices_;
  using Base::adjacencyList_;

  using thisType = DependencyGraphAccessesT<VertexData>;

  DependencyGraphAccessesT<VertexData>(StencilInstantiation* stencilInstantiation)
      : Base(), instantiation_(stencilInstantiation) {}

  /// @brief Construct graph by merging the given `graphs`
  ///
  /// @param graphs       Graphs to merge
  /// @tparam GraphTypes  Varidaic pack of `std::shared_ptr<DependencyGraphAccesses>`
  template <class... GraphTypes>
  DependencyGraphAccessesT(StencilInstantiation* stencilInstantiation, const GraphTypes&... graphs)
      : DependencyGraphAccessesT(stencilInstantiation) {
    static_assert(
        and_<std::is_same<GraphTypes, std::shared_ptr<thisType>>...>::value,
        "GraphTypes needs to be a varidaic pack of `std::shared_ptr<DependencyGraphAccesses>`");
    for(const auto& g : {graphs...})
      merge(g.get());
  }

  /// @brief Process the StatementAccessPair and insert it into the current graph
  ///
  /// For each write and read access a node will be inserted. Between each write and read access an
  /// edge will be created s.t
  ///
  /// +-------+           +--------+
  /// | Write | --------> |  Read  |
  /// +-------+           +--------+
  ///
  /// Note that only child-less nodes are processed.
  void insertStatementAccessesPair(const std::shared_ptr<StatementAccessesPair>& stmtAccessPair) {
    if(stmtAccessPair->hasChildren()) {
      for(const auto& s : stmtAccessPair->getChildren())
        insertStatementAccessesPair(s);
    } else {

      for(const auto& writeAccess : stmtAccessPair->getAccesses()->getWriteAccesses()) {
        insertNode(writeAccess.first);

        for(const auto& readAccess : stmtAccessPair->getAccesses()->getReadAccesses())
          Base::insertEdge(writeAccess.first, readAccess.first, readAccess.second);
      }
    }
  }

  /// @brief Insert a new node
  template <typename Int>
  Vertex& insertNode(Int ID,
                     typename std::enable_if<vertex_has_data<Int, VertexData>::value>::type* = 0) {
    Vertex& vertex = Base::insertNode(ID);
    VertexIDToAccessIDMap_.emplace(vertex.VertexID, vertex.ID);
    return vertex;
  }

  template <typename Int>
  Vertex&
  insertNode(Int ID, VertexDataEmul data,
             typename std::enable_if<(!vertex_has_data<Int, VertexData>::value)>::type* = 0) {
    Vertex& vertex = Base::insertNode(ID, data);
    VertexIDToAccessIDMap_.emplace(vertex.VertexID, vertex.ID);
    return vertex;
  }

  /// @brief Merge extents if edge already exists
  void edgeAlreadyExists(EdgeData& existingEdge, const EdgeData& newEdge) {
    if(!newEdge.isPointwise())
      existingEdge.merge(newEdge);
  }

  /// @brief Get the ID of the vertex given by ID
  int getIDFromVertexID(std::size_t VertexID) const {
    auto it = VertexIDToAccessIDMap_.find(VertexID);
    DAWN_ASSERT_MSG(it != VertexIDToAccessIDMap_.end(), "Node with given VertexID does not exist");
    return it->second;
  }

  /// @brief EdgeData to string
  const char* edgeDataToString(const EdgeData& data) const {
    if(data.isHorizontalPointwise() && data.isVerticalPointwise())
      return " -------> ";
    else
      return "\033[1;37m ~~~~~~~> \033[0m";
  }

  /// @brief EdgeData to dot
  std::string edgeDataToDot(const EdgeData& data) const {
    if(data.isHorizontalPointwise() && data.isVerticalPointwise())
      return " [style = dashed]";
    else
      return RangeToString(", ", " [label = \"<",
                           ">\"]")(data.getExtents(), [](const Extent& extent) {
        return std::to_string(extent.Minus) + ", " + std::to_string(extent.Plus);
      });
  }

  /// @brief Get the shape of the dot grahps
  const char* getDotShape() const { return "circle"; }

  /// @brief Get the name of the vertex given by ID
  std::string getVertexNameByVertexID(std::size_t VertexID) const {
    return instantiation_ ? instantiation_->getNameFromAccessID(getIDFromVertexID(VertexID))
                          : std::to_string(getIDFromVertexID(VertexID));
  }

  /// @brief Merge `other` into `this`
  void merge(const thisType* other) {
    // Insert the nodes of `other`
    for(const auto& AccessIDVertexIDPair : other->getVertices()) {
      int AccessID = AccessIDVertexIDPair.first;
      insertNode(AccessID);
    }

    // Insert the edges of `other`
    for(std::size_t VertexID = 0; VertexID < other->getAdjacencyList().size(); ++VertexID) {
      for(const Edge& edge : *(other->getAdjacencyList()[VertexID])) {
        Base::insertEdge(other->getIDFromVertexID(edge.FromVertexID),
                         other->getIDFromVertexID(edge.ToVertexID), edge.Data);
      }
    }
  }

  /// @brief Clone the graph by performing a @b deep copy
  std::shared_ptr<thisType> clone() const {
    auto graph = std::make_shared<DependencyGraphAccesses>(instantiation_);
    graph->vertices_ = vertices_;
    graph->VertexIDToAccessIDMap_ = VertexIDToAccessIDMap_;
    for(const auto& edgeListPtr : adjacencyList_)
      graph->adjacencyList_.push_back(std::make_shared<EdgeList>(*edgeListPtr));
    return graph;
  }

  /// @brief Partition the graph into the non-connected sub-graphs
  std::vector<std::set<std::size_t>> partitionInSubGraphs() const {

    // Each Vertex is assigned to a partition (-1 means not yet assigned)
    std::vector<int> partition(adjacencyList_.size(), -1);
    int numPartitions = 0;

    std::vector<std::size_t> nodesToVisit;

    // Perform a BFS
    for(std::size_t VertexID = 0; VertexID < adjacencyList_.size(); ++VertexID) {
      int currentPartitionIdx = partition[VertexID];
      if(currentPartitionIdx != -1)
        continue;

      // Create a new partition
      currentPartitionIdx = numPartitions++;

      nodesToVisit.push_back(VertexID);
      while(!nodesToVisit.empty()) {

        // Process next node
        std::size_t curNode = nodesToVisit.back();
        nodesToVisit.pop_back();

        int nodePartitionIdx = partition[curNode];

        // If we already visited this node, we have to merge our current partition (given by
        // `currentPartitionIdx`) with the other partition already visited.
        if(nodePartitionIdx != -1) {
          if(nodePartitionIdx == currentPartitionIdx)
            continue;

          int otherPartitionIdx = nodePartitionIdx;
          DAWN_ASSERT(otherPartitionIdx != -1);

          // Merge the current partition into other
          std::for_each(partition.begin(), partition.end(), [&](int& partitionIdx) {
            if(partitionIdx == currentPartitionIdx)
              partitionIdx = otherPartitionIdx;
          });

          // Current partition is now equal to the other partition
          currentPartitionIdx = otherPartitionIdx;
          continue;
        } else {
          partition[curNode] = currentPartitionIdx;
        }

        for(const Edge& edge : *adjacencyList_[curNode])
          nodesToVisit.push_back(edge.ToVertexID);
      }
    }

    // Assemble the final partitions. We use a map between the index in `finalPartitions` and the
    // partitionIdx.
    std::vector<std::set<std::size_t>> finalPartitions;
    std::unordered_map<int, std::size_t> PartitionIndexToIndexInFinalPartitionsMap;

    for(std::size_t VertexID = 0; VertexID < adjacencyList_.size(); ++VertexID) {
      int partitionIdx = partition[VertexID];
      auto it = PartitionIndexToIndexInFinalPartitionsMap.find(partitionIdx);

      if(it != PartitionIndexToIndexInFinalPartitionsMap.end())
        finalPartitions[it->second].insert(VertexID);
      else {
        PartitionIndexToIndexInFinalPartitionsMap.emplace(partitionIdx, finalPartitions.size());
        finalPartitions.push_back(std::set<std::size_t>{});
        finalPartitions.back().insert(VertexID);
      }
    }
    return finalPartitions;
  }

  /// @brief Check if the graph is a DAG
  ///
  /// If the graph consists of non-connected subgraphs, it will check that each subgraph is a DAG.
  /// In our context, a DAG is defined as having a non-empty set of input as well as output nodes.
  bool isDAG() const {
    auto partitions = partitionInSubGraphs();
    std::vector<std::size_t> vertices;

    for(std::set<std::size_t>& partition : partitions) {
      vertices.empty();
      getInputVertexIDsImpl(partition, [](std::size_t VertexID) { return VertexID; }, vertices);
      if(vertices.empty())
        return false;

      vertices.clear();
      getOutputVertexIDsImpl(partition, [](std::size_t VertexID) { return VertexID; }, vertices);
      if(vertices.empty())
        return false;
    }
    return true;
  }

  /// @brief Get the VertexIDs of the pure `output` vertices
  ///
  /// Output vertices are vertices which do not have incoming edges from other vertices.
  std::vector<std::size_t> getOutputVertexIDs() const {
    std::vector<std::size_t> outputVertexIDs;
    getOutputVertexIDsImpl(
        vertices_,
        [](const std::pair<int, Vertex>& IDVertexPair) { return IDVertexPair.second.VertexID; },
        outputVertexIDs);
    return outputVertexIDs;
  }

  /// @brief Generic version of computing the Input-VertexIDs
  ///
  /// This function can operate on the vertex list of the graph as well as on a simple set of
  /// VertexIDs.
  template <class VertexListType, class GetVertexIDFromVertexListElemenFuncType>
  void getInputVertexIDsImpl(
      const VertexListType& vertexList,
      GetVertexIDFromVertexListElemenFuncType&& getVertexIDFromVertexListElemenFunc,
      std::vector<std::size_t>& inputVertexIDs) const {

    const auto& adjacencyList = Base::getAdjacencyList();
    for(const auto& vertex : vertexList) {
      std::size_t VertexID = getVertexIDFromVertexListElemenFunc(vertex);
      if(adjacencyList[VertexID]->empty())
        inputVertexIDs.push_back(VertexID);
      else if(adjacencyList[VertexID]->size() == 1) {
        // We allow self-dependencies!
        const auto& edge = adjacencyList[VertexID]->front();
        if(edge.FromVertexID == edge.ToVertexID)
          inputVertexIDs.push_back(VertexID);
      }
    }
  }

  /// @brief Generic version of computing the Output-VertexIDs
  ///
  /// This function can operate on the vertex list of the graph as well as on a simple set of
  /// VertexIDs.
  template <class VertexListType, class GetVertexIDFromVertexListElemenFuncType>
  void getOutputVertexIDsImpl(
      const VertexListType& vertexList,
      GetVertexIDFromVertexListElemenFuncType&& getVertexIDFromVertexListElemenFunc,
      std::vector<std::size_t>& outputVertexIDs) const {

    std::set<std::size_t> dependentNodes;
    const auto& adjacencyList = Base::getAdjacencyList();

    // Construct a set of dependent nodes i.e nodes with edges from other nodes pointing to them
    for(const auto& edgeList : adjacencyList)
      for(const auto& edge : *edgeList)
        // We allow self-dependencies!
        if(edge.FromVertexID != edge.ToVertexID)
          dependentNodes.insert(edge.ToVertexID);

    for(const auto& vertex : vertexList) {
      std::size_t VertexID = getVertexIDFromVertexListElemenFunc(vertex);
      if(!dependentNodes.count(VertexID))
        outputVertexIDs.push_back(VertexID);
    }
  }

  /// @brief Get the VertexIDs of the pure `input` vertices
  ///
  /// Input vertices are vertices which only have outgoing edges i.e no incoming edges. This will
  /// almost always contain all the literal accesses.
  std::vector<std::size_t> getInputVertexIDs() const {
    std::vector<std::size_t> inputVertexIDs;
    getInputVertexIDsImpl(
        vertices_,
        [](const std::pair<int, Vertex>& IDVertexPair) { return IDVertexPair.second.VertexID; },
        inputVertexIDs);
    return inputVertexIDs;
  }

  /// @brief Find @b multi-node strongly connected components (SCC) in the accesses graph using
  /// Tarjan's algorithm
  ///
  /// @param scc      The output set of multi-node strongly connected components (given as a vector
  ///                 of vectors of AccessIDs). The vector needs to be allocated but will be cleared
  ///                 upon invocation.
  /// @returns `true` if atleast one multi-node SCC was found, `false` otherwise.
  ///
  /// @see https://en.wikipedia.org/wiki/Tarjan's_strongly_connected_components_algorithm
  bool findStronglyConnectedComponents(std::vector<std::set<int>>& scc) const {
    return StronglyConnectedComponents(this, &scc).find();
  }

  /// @brief Check if the accesses graph contains any @b multi-node strongly connected components
  /// using Tarjan's algorithm
  ///
  /// @returns `true` if atleast one multi-node SCC was found, `false` otherwise.
  ///
  /// @see https://en.wikipedia.org/wiki/Tarjan's_strongly_connected_components_algorithm
  bool hasStronglyConnectedComponents() const {
    return StronglyConnectedComponents(this, nullptr).find();
  }

  /// @brief Greedily colors the graph
  ///
  /// @param coloring   Map of AccessID to assigned color
  ///
  /// @see https://en.wikipedia.org/wiki/Greedy_coloring
  void greedyColoring(std::unordered_map<int, int>& coloring) const {
    return GreedyColoring(this, coloring).compute();
  }

  /// @brief Clear the graph
  void clear() {
    Base::clear();
    VertexIDToAccessIDMap_.clear();
  }

  /// @brief Get stencil instantiation
  StencilInstantiation* getStencilInstantiation() const { return instantiation_; }

  /// @brief Serialize the graph to JSON
  void toJSON(const std::string& file) const {
    std::unordered_map<std::size_t, Extents> extentMap = *computeBoundaryExtents(this);
    json::json jgraph;

    auto extentsToVec = [&](const Extents& extents) {
      std::vector<int> extentsVec;
      for(const Extent& extent : extents.getExtents()) {
        extentsVec.push_back(extent.Minus);
        extentsVec.push_back(extent.Plus);
      }
      return extentsVec;
    };

    std::size_t EdgeID = 0;
    for(std::size_t VertexID = 0; VertexID < Base::getNumVertices(); ++VertexID) {
      json::json jvertex;

      jvertex["name"] = getVertexNameByVertexID(VertexID);
      jvertex["extent"] = extentsToVec(extentMap[VertexID]);

      int AccessID = getIDFromVertexID(VertexID);
      if(instantiation_->isTemporaryField(AccessID))
        jvertex["type"] = "field_temporary";
      else if(instantiation_->isField(AccessID))
        jvertex["type"] = "field";
      else if(instantiation_->isVariable(AccessID) || instantiation_->isGlobalVariable(AccessID))
        jvertex["type"] = "variable";
      else if(instantiation_->isLiteral(AccessID))
        jvertex["type"] = "literal";
      else
        dawn_unreachable("invalid vertex type");

      jgraph["vertices"][std::to_string(VertexID)] = jvertex;

      for(const Edge& edge : *Base::getAdjacencyList()[VertexID]) {
        json::json jedge;

        jedge["from"] = edge.FromVertexID;
        jedge["to"] = edge.ToVertexID;
        jedge["extent"] = extentsToVec(edge.Data);
        jgraph["edges"][std::to_string(EdgeID)] = jedge;

        EdgeID++;
      }
    }

    jgraph["num_vertices"] = Base::getNumVertices();
    jgraph["num_edges"] = EdgeID;

    std::ofstream ofs;
    ofs.open(file);
    if(!ofs.is_open()) {
      DiagnosticsBuilder diag(DiagnosticsKind::Error);
      diag << "failed to open file: \"" << file << "\"";
      instantiation_->getOptimizerContext()->getDiagnostics().report(diag);
    }

    ofs << jgraph.dump(2);
    ofs.close();
  }
};

using DependencyGraphAccesses = DependencyGraphAccessesT<void>;

} // namespace dawn

#endif
