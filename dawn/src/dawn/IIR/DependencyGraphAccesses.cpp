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

#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/IIR/ASTStmt.h"
#include "dawn/IIR/StencilMetaInformation.h"
#include "dawn/Support/Exception.h"
#include "dawn/Support/Json.h"
#include "dawn/Support/Logger.h"
#include "dawn/Support/StringUtil.h"
#include <stack>
#include <unordered_map>

namespace dawn {
namespace iir {

void DependencyGraphAccesses::insertStatement(const std::shared_ptr<iir::Stmt>& stmt) {

  if(!stmt->getChildren().empty()) {
    for(const auto& s : stmt->getChildren())
      insertStatement(s);
  } else {
    const auto& callerAccesses = stmt->getData<iir::IIRStmtData>().CallerAccesses;

    for(const auto& writeAccess : callerAccesses->getWriteAccesses()) {
      insertNode(writeAccess.first);

      for(const auto& readAccess : callerAccesses->getReadAccesses())
        insertEdge(writeAccess.first, readAccess.first, readAccess.second);
    }
  }
}

DependencyGraphAccesses::Vertex& DependencyGraphAccesses::insertNode(int ID) {
  Vertex& vertex = Base::insertNode(ID);
  VertexIDToAccessIDMap_.emplace(vertex.VertexID, vertex.Value);
  return vertex;
}

void DependencyGraphAccesses::edgeAlreadyExists(DependencyGraphAccesses::EdgeData& existingEdge,
                                                const DependencyGraphAccesses::EdgeData& newEdge) {
  if(!newEdge.isPointwise())
    existingEdge.merge(newEdge);
}

int DependencyGraphAccesses::getIDFromVertexID(std::size_t VertexID) const {
  auto it = VertexIDToAccessIDMap_.find(VertexID);
  DAWN_ASSERT_MSG(it != VertexIDToAccessIDMap_.end(), "Node with given VertexID does not exist");
  return it->second;
}

const char* DependencyGraphAccesses::edgeDataToString(const EdgeData& data) const {
  if(data.isHorizontalPointwise() && data.isVerticalPointwise())
    return " -------> ";
  else
    return "\033[1;37m ~~~~~~~> \033[0m";
}

std::string DependencyGraphAccesses::edgeDataToDot(const EdgeData& data) const {
  if(data.isHorizontalPointwise() && data.isVerticalPointwise())
    return " [style = dashed]";
  else {
    const auto& hExtents = extent_cast<CartesianExtent const&>(data.horizontalExtent());
    const auto& vExtents = data.verticalExtent();
    return " [label = \"<" + std::to_string(hExtents.iMinus()) + ", " +
           std::to_string(hExtents.iPlus()) + ", " + std::to_string(hExtents.jMinus()) + ", " +
           std::to_string(hExtents.jPlus()) + ", " + std::to_string(vExtents.minus()) + ", " +
           std::to_string(vExtents.plus()) + ">\"]";
  }
}

const char* DependencyGraphAccesses::getDotShape() const { return "circle"; }

std::string DependencyGraphAccesses::getVertexNameByVertexID(std::size_t VertexID) const {
  return metaData_.get().getFieldNameFromAccessID(getIDFromVertexID(VertexID));
}

void DependencyGraphAccesses::merge(const DependencyGraphAccesses& other) {
  // Insert the nodes of `other`
  for(const auto& AccessIDVertexIDPair : other.getVertices()) {
    int AccessID = AccessIDVertexIDPair.first;
    insertNode(AccessID);
  }

  // Insert the edges of `other`
  for(std::size_t VertexID = 0; VertexID < other.getAdjacencyList().size(); ++VertexID) {
    for(const Edge& edge : other.getAdjacencyList()[VertexID]) {
      insertEdge(other.getIDFromVertexID(edge.FromVertexID),
                 other.getIDFromVertexID(edge.ToVertexID), edge.Data);
    }
  }
}

std::vector<std::set<std::size_t>> DependencyGraphAccesses::partitionInSubGraphs() const {

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

      for(const Edge& edge : adjacencyList_[curNode])
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

/// @brief Generic version of computing the Input-VertexIDs
///
/// This function can operate on the vertex list of the graph as well as on a simple set of
/// VertexIDs.
template <class VertexListType, class GetVertexIDFromVertexListElemenFuncType>
void getInputVertexIDsImpl(
    const DependencyGraphAccesses& graph, const VertexListType& vertexList,
    GetVertexIDFromVertexListElemenFuncType&& getVertexIDFromVertexListElemenFunc,
    std::vector<std::size_t>& inputVertexIDs) {

  const auto& adjacencyList = graph.getAdjacencyList();
  for(const auto& vertex : vertexList) {
    std::size_t VertexID = getVertexIDFromVertexListElemenFunc(vertex);
    if(adjacencyList[VertexID].empty())
      inputVertexIDs.push_back(VertexID);
    else if(adjacencyList[VertexID].size() == 1) {
      // We allow self-dependencies!
      const auto& edge = adjacencyList[VertexID].front();
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
    const DependencyGraphAccesses& graph, const VertexListType& vertexList,
    GetVertexIDFromVertexListElemenFuncType&& getVertexIDFromVertexListElemenFunc,
    std::vector<std::size_t>& outputVertexIDs) {

  std::set<std::size_t> dependentNodes;
  const auto& adjacencyList = graph.getAdjacencyList();

  // Construct a set of dependent nodes i.e nodes with edges from other nodes pointing to them
  for(const auto& edgeList : adjacencyList)
    for(const auto& edge : edgeList)
      // We allow self-dependencies!
      if(edge.FromVertexID != edge.ToVertexID)
        dependentNodes.insert(edge.ToVertexID);

  for(const auto& vertex : vertexList) {
    std::size_t VertexID = getVertexIDFromVertexListElemenFunc(vertex);
    if(!dependentNodes.count(VertexID))
      outputVertexIDs.push_back(VertexID);
  }
}

bool DependencyGraphAccesses::isDAG() const {
  auto partitions = partitionInSubGraphs();
  std::vector<std::size_t> vertices;

  for(std::set<std::size_t>& partition : partitions) {
    getInputVertexIDsImpl(
        *this, partition, [](std::size_t VertexID) { return VertexID; }, vertices);
    if(vertices.empty())
      return false;

    vertices.clear();
    getOutputVertexIDsImpl(
        *this, partition, [](std::size_t VertexID) { return VertexID; }, vertices);
    if(vertices.empty())
      return false;
  }
  return true;
}

std::vector<std::size_t> DependencyGraphAccesses::getOutputVertexIDs() const {
  std::vector<std::size_t> outputVertexIDs;
  getOutputVertexIDsImpl(
      *this, vertices_,
      [](const std::pair<int, Vertex>& IDVertexPair) { return IDVertexPair.second.VertexID; },
      outputVertexIDs);
  return outputVertexIDs;
}

std::vector<std::size_t> DependencyGraphAccesses::getInputVertexIDs() const {
  std::vector<std::size_t> inputVertexIDs;
  getInputVertexIDsImpl(
      *this, vertices_,
      [](const std::pair<int, Vertex>& IDVertexPair) { return IDVertexPair.second.VertexID; },
      inputVertexIDs);
  return inputVertexIDs;
}

/// @fn computeBoundaryPoints
/// @brief Compute the accumulated extent of each Vertex (given by `VertexID`) referenced in `graph`
/// @returns map of `VertexID` to boundary extent
/// @ingroup optimizer
static std::unordered_map<std::size_t, iir::Extents>
computeBoundaryExtents(const iir::DependencyGraphAccesses* graph) {
  using Vertex = iir::DependencyGraphAccesses::Vertex;
  using Edge = iir::DependencyGraphAccesses::Edge;

  const auto& adjacencyList = graph->getAdjacencyList();

  std::vector<std::size_t> nodesToVisit;
  std::unordered_set<std::size_t> visitedNodes;

  // Keep track of the extents of each vertex (and compute a VertexID to AccessID map)
  std::unordered_map<std::size_t, iir::Extents> nodeExtents;

  for(const auto& AccessIDVertexPair : graph->getVertices()) {
    const Vertex& vertex = AccessIDVertexPair.second;
    nodeExtents.emplace(vertex.VertexID, iir::Extents{});
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
      const iir::Extents& curExtent = nodeExtents.at(curNode);

      // Check if we already visited this node
      if(visitedNodes.count(curNode))
        continue;
      else
        visitedNodes.insert(curNode);

      // Follow edges of the current node and update the node extents
      for(const Edge& edge : adjacencyList[curNode]) {
        nodeExtents.at(edge.ToVertexID).merge(curExtent + edge.Data);
        nodesToVisit.push_back(edge.ToVertexID);
      }
    }
  }

  return nodeExtents;
}

namespace {

/// @brief Find strongly connected components using Tarjan's algorithm
///
/// @see https://en.wikipedia.org/wiki/Tarjan's_strongly_connected_components_algorithm
class StronglyConnectedComponents {
  using VertexType = DependencyGraphAccesses::Vertex;
  using EdgeType = DependencyGraphAccesses::Edge;

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
    const VertexType* Vertex;
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
  bool find() {
    if(scc_)
      scc_->clear();

    // Initialize the vertex Data
    for(const auto& AccessIDVertexPair : graph_->getVertices()) {
      const VertexType& vertex = AccessIDVertexPair.second;
      vertexData_.emplace(vertex.VertexID, VertexData{-1, -1, false, &vertex});
    }
    index_ = 0;
    hasMultiNodeSCC_ = false;

    // Iterate all Vertices
    for(const auto& AccessIDVertexPair : graph_->getVertices()) {
      const VertexType& vertex = AccessIDVertexPair.second;
      findImpl(vertex.VertexID);
    }

    return hasMultiNodeSCC_;
  }

private:
  void findImpl(std::size_t FromVertexID) {
    VertexData& FromVertexData = vertexData_[FromVertexID];

    if(FromVertexData.Index != -1)
      return;

    // Set the depth index for the `FromVertex` to the smallest unused index
    FromVertexData.Index = index_;
    FromVertexData.LowLink = index_;
    FromVertexData.OnStack = true;
    vertexStack_.push(FromVertexID);
    index_++;

    // Consider successors of the `FromVertex`
    for(const EdgeType& edge : graph_->getAdjacencyList()[FromVertexID]) {

      VertexData& ToVertexData = vertexData_[edge.ToVertexID];

      if(ToVertexData.Index == -1) {
        // Successor `ToVertex` has not yet been visited; recurse on it
        findImpl(edge.ToVertexID);
        FromVertexData.LowLink = std::min(FromVertexData.LowLink, ToVertexData.LowLink);
      } else if(ToVertexData.OnStack) {
        // Successor `ToVertex` is in stack and hence in the current SCC
        FromVertexData.LowLink = std::min(FromVertexData.LowLink, ToVertexData.LowLink);
      }
    }

    // If `FromVertex` is a root node, pop the stack and generate an SCC
    if(FromVertexData.LowLink == FromVertexData.Index) {
      std::set<int> SCC;
      std::size_t VertexID;

      do {
        VertexID = vertexStack_.top();
        vertexStack_.pop();

        VertexData& vertexData = vertexData_[VertexID];
        vertexData.OnStack = false;
        SCC.insert(vertexData.Vertex->Value);

      } while(VertexID != FromVertexID);

      if(SCC.size() > 1) {
        hasMultiNodeSCC_ = true;
        if(scc_)
          scc_->emplace_back(std::move(SCC));
      }
    }
  }
};

} // anonymous namespace

bool DependencyGraphAccesses::findStronglyConnectedComponents(
    std::vector<std::set<int>>& scc) const {
  return StronglyConnectedComponents(this, &scc).find();
}

bool DependencyGraphAccesses::hasStronglyConnectedComponents() const {
  return StronglyConnectedComponents(this, nullptr).find();
}

namespace {

/// @brief Greedily color the graph
class GreedyColoring {
  const DependencyGraphAccesses* graph_;
  std::unordered_map<int, int>& coloring_;

public:
  GreedyColoring(const DependencyGraphAccesses* graph, std::unordered_map<int, int>& coloring)
      : graph_(graph), coloring_(coloring) {}

  using Vertex = DependencyGraphAccesses::Vertex;
  using Edge = DependencyGraphAccesses::Edge;

  void compute() {
    coloring_.clear();

    std::size_t numVertices = graph_->getNumVertices();
    const auto& adjacencyList = graph_->getAdjacencyList();

    // Compute the neighbor-list
    std::vector<std::set<std::size_t>> neighborList(numVertices);
    for(std::size_t FromVertexID = 0; FromVertexID < numVertices; ++FromVertexID) {
      for(const Edge& edge : adjacencyList[FromVertexID]) {
        neighborList[edge.FromVertexID].insert(edge.ToVertexID);
        neighborList[edge.ToVertexID].insert(edge.FromVertexID);
      }
    }

    // True value of `available[VertexID]` means that the color `VertexID` is assigned to one of
    // its adjacent vertices
    std::vector<bool> assigned(numVertices, false);

    // Colors of the VertexID
    std::vector<int> colors(numVertices, -1);
    colors[0] = 0;

    // Assign colors to the remaining vertices
    for(std::size_t FromVertexID = 1; FromVertexID < numVertices; ++FromVertexID) {

      auto setAssignmentOfNeigbors = [&](bool isAssigned) {
        for(std::size_t ToVertexID : neighborList[FromVertexID])
          if(colors[ToVertexID] != -1)
            assigned[colors[ToVertexID]] = isAssigned;
      };

      // Process all neighbor vertices and flag their colors as assigned
      setAssignmentOfNeigbors(true);

      // Find the first available color
      auto it = std::find(assigned.begin(), assigned.end(), false);
      colors[FromVertexID] =
          it == assigned.end() ? assigned.size() - 1 : std::distance(assigned.begin(), it);

      // Reset the values back to false for the next iteration
      setAssignmentOfNeigbors(false);
    }

    for(std::size_t VertexID = 0; VertexID < numVertices; ++VertexID)
      coloring_.emplace(graph_->getIDFromVertexID(VertexID), colors[VertexID]);
  }
};

} // anonymous namespace

void DependencyGraphAccesses::greedyColoring(std::unordered_map<int, int>& coloring) const {
  return GreedyColoring(this, coloring).compute();
}

void DependencyGraphAccesses::clear() {
  Base::clear();
  VertexIDToAccessIDMap_.clear();
}

void DependencyGraphAccesses::toJSON(const std::string& file) const {
  StencilMetaInformation const& metaData = metaData_;

  std::unordered_map<std::size_t, Extents> extentMap = computeBoundaryExtents(this);
  json::json jgraph;

  auto extentsToVec = [&](const Extents& extents) {
    std::vector<int> extentsVec;

    auto vExtent = extents.verticalExtent();
    auto hExtent = extent_cast<CartesianExtent const&>(extents.horizontalExtent());

    extentsVec.push_back(hExtent.iMinus());
    extentsVec.push_back(hExtent.iPlus());
    extentsVec.push_back(hExtent.jMinus());
    extentsVec.push_back(hExtent.jPlus());
    extentsVec.push_back(vExtent.minus());
    extentsVec.push_back(vExtent.plus());

    return extentsVec;
  };

  std::size_t EdgeID = 0;
  for(std::size_t VertexID = 0; VertexID < getNumVertices(); ++VertexID) {
    json::json jvertex;

    jvertex["name"] = getVertexNameByVertexID(VertexID);
    jvertex["extent"] = extentsToVec(extentMap.at(VertexID));

    int AccessID = getIDFromVertexID(VertexID);
    if(metaData.isAccessType(iir::FieldAccessType::StencilTemporary, AccessID))
      jvertex["type"] = "field_temporary";
    else if(metaData.isAccessType(FieldAccessType::Field, AccessID))
      jvertex["type"] = "field";
    else if(metaData.isAccessType(FieldAccessType::LocalVariable, AccessID) ||
            metaData.isAccessType(iir::FieldAccessType::GlobalVariable, AccessID))
      jvertex["type"] = "variable";
    else if(metaData.isAccessType(iir::FieldAccessType::Literal, AccessID))
      jvertex["type"] = "literal";
    else
      dawn_unreachable("invalid vertex type");

    jgraph["vertices"][std::to_string(VertexID)] = jvertex;

    for(const Edge& edge : getAdjacencyList()[VertexID]) {
      json::json jedge;

      jedge["from"] = edge.FromVertexID;
      jedge["to"] = edge.ToVertexID;
      jedge["extent"] = extentsToVec(edge.Data);
      jgraph["edges"][std::to_string(EdgeID)] = jedge;

      EdgeID++;
    }
  }

  jgraph["num_vertices"] = getNumVertices();
  jgraph["num_edges"] = EdgeID;

  // TODO deal with it w/o stencilInstantiation
  std::ofstream ofs;
  ofs.open(file);
  if(!ofs.is_open()) {
    throw CompileError(std::string("Failed to open file: ") + file);
  }

  ofs << jgraph.dump(2);
  ofs.close();
}

bool DependencyGraphAccesses::exceedsMaxBoundaryPoints(int maxHorizontalBoundaryExtent) {
  std::unordered_map<std::size_t, Extents> extentMap = computeBoundaryExtents(this);

  for(const auto& vertexIDExtentsPair : extentMap) {
    auto const& hExtent =
        extent_cast<iir::CartesianExtent const&>(vertexIDExtentsPair.second.horizontalExtent());
    if(hExtent.iPlus() > maxHorizontalBoundaryExtent ||
       hExtent.iMinus() < -maxHorizontalBoundaryExtent ||
       hExtent.jPlus() > maxHorizontalBoundaryExtent ||
       hExtent.jMinus() < -maxHorizontalBoundaryExtent)
      return true;
  }
  return false;
}

} // namespace iir
} // namespace dawn
