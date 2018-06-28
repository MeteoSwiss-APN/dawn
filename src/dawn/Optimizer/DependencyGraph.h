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

#ifndef DAWN_OPTIMIZER_DEPENDENCYGRAPH_H
#define DAWN_OPTIMIZER_DEPENDENCYGRAPH_H

#include "dawn/Support/Assert.h"
#include "dawn/Support/Unreachable.h"
#include <algorithm>
#include <fstream>
#include <iosfwd>
#include <iterator>
#include <list>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dawn {

/// @brief CRTP base class of all dependency graphs
/// @ingroup optimizer
template <class Derived, class EdgeData>
class DependencyGraph {
public:
  /// @brief Directed edge between the vertex `From` and `To` containing `Data`
  ///
  /// @verbatim
  ///               Data
  ///    From  -------------> To
  ///
  /// @endverbatim
  struct Edge {
    EdgeData Data;            ///< What connection does the edge represent
    std::size_t FromVertexID; ///< Vertex ID of `From`
    std::size_t ToVertexID;   ///< Vertex ID of `To`

    bool operator==(const Edge& other) const {
      return (Data == other.Data) && (ToVertexID == other.ToVertexID) &&
             (FromVertexID == other.FromVertexID);
    }
    bool operator!=(const Edge& other) const { return !(*this == other); }
  };

  using EdgeList = std::list<Edge>;

  struct Vertex {
    std::size_t VertexID; ///< Unqiue ID of the Vertex
    int ID;               ///< ID of the data to be stored
  };

  /// @brief Get the adjacency list
  /// @{
  std::vector<std::shared_ptr<EdgeList>>& getAdjacencyList() { return adjacencyList_; }
  const std::vector<std::shared_ptr<EdgeList>>& getAdjacencyList() const { return adjacencyList_; }
  /// @}

  /// @brief Get the vertices
  /// @{
  std::unordered_map<int, Vertex>& getVertices() { return vertices_; }
  const std::unordered_map<int, Vertex>& getVertices() const { return vertices_; }
  /// @}

  //===----------------------------------------------------------------------------------------===//
  //     Graph implementation
  //===----------------------------------------------------------------------------------------===//

  /// @brief Initialize empty graph
  DependencyGraph() = default;

  /// @brief Insert a new node
  Vertex& insertNode(int ID) {
    auto insertPair = vertices_.emplace(ID, Vertex{adjacencyList_.size(), ID});
    if(insertPair.second)
      adjacencyList_.push_back(std::make_shared<EdgeList>());
    return insertPair.first->second;
  }

  /// @brief Insert a new edge from `IDFrom` to `IDTo` containing `data`
  ///
  /// @verbatim
  ///                    EdgeData
  ///      X  ------------------------------- X
  ///    IDFrom                              IDTo
  /// @endverbatim
  template <typename TEdgeData>
  void insertEdge(int IDFrom, int IDTo, TEdgeData&& data) {

    // Create `IDTo` node (We shift the burden to the `insertNode` to take appropriate actions
    // if the node does already exist)
    static_cast<Derived*>(this)->insertNode(IDTo);

    // Traverse the edge-list of node `IDFrom` to check if we already have such an edge
    auto& edgeList = adjacencyList_[getVertexIDFromID(IDFrom)];
    const Edge edge{std::forward<TEdgeData>(data), getVertexIDFromID(IDFrom),
                    getVertexIDFromID(IDTo)};

    auto it = std::find_if(edgeList->begin(), edgeList->end(), [&edge](const Edge& e) {
      return e.FromVertexID == edge.FromVertexID && e.ToVertexID == edge.ToVertexID;
    });

    if(it != edgeList->end())
      static_cast<Derived*>(this)->edgeAlreadyExists(it->Data, edge.Data);
    else
      edgeList->push_back(edge);
  }

  /// @brief Callback which will be invoked if an edge already exists
  ///
  /// This is useful for access graph which want to merge the extents (= EdgeData).
  void edgeAlreadyExists(EdgeData& existingEdge, const EdgeData& newEdge) {}

  /// @brief Get the ID of the Vertex in the graph given an ID
  std::size_t getVertexIDFromID(int ID) const {
    auto it = vertices_.find(ID);
    DAWN_ASSERT_MSG(it != vertices_.end(), "Node with given ID does not exist");
    return it->second.VertexID;
  }

  /// @brief Get the ID of the vertex given by ID
  int getIDFromVertexID(std::size_t VertexID) const {
    for(const auto& vertexPair : vertices_)
      if(vertexPair.second.VertexID == VertexID)
        return vertexPair.first;
    dawn_unreachable("invalid VertexID");
  }

  /// @brief Get the list of edges of node given by `ID`
  EdgeList& edgesOf(int ID) { return *(adjacencyList_[getVertexIDFromID(ID)]); }
  const EdgeList& edgesOf(int ID) const { return *(adjacencyList_[getVertexIDFromID(ID)]); }

  /// @brief Clear the graph
  void clear() {
    vertices_.clear();
    adjacencyList_.clear();
  }

  /// @brief Check if graph is empty
  bool empty() const { return vertices_.empty(); }

  /// @brief Get number of vertices
  std::size_t getNumVertices() const { return vertices_.size(); }

  //===----------------------------------------------------------------------------------------===//
  //     Graph visualization
  //===----------------------------------------------------------------------------------------===//

  /// @brief Write graph to `.dot` file
  void toDot(std::string filename) const {
    std::ofstream fout(filename);
    DAWN_ASSERT(fout.is_open());
    toDotImpl(fout);
    fout.close();
  }
  std::string toDot() const {
    std::stringstream ss;
    toDotImpl(ss);
    return ss.str();
  }

  /// @brief Convert graph to string
  std::string toString() const {
    std::stringstream ss;
    for(std::size_t VertexID = 0; VertexID < adjacencyList_.size(); ++VertexID) {
      for(const Edge& edge : *(adjacencyList_[VertexID])) {
        ss << static_cast<const Derived*>(this)->getVertexNameByVertexID(edge.FromVertexID)
           << static_cast<const Derived*>(this)->edgeDataToString(edge.Data)
           << static_cast<const Derived*>(this)->getVertexNameByVertexID(edge.ToVertexID) << "\n";
      }
    }
    return ss.str();
  }

  /// @brief Stream graph
  template <class StreamType>
  friend std::ostream& operator<<(StreamType& os, const DependencyGraph& graph) {
    return (os << graph.toString());
  }

protected:
  template <class StreamType>
  void toDotImpl(StreamType& os) const {
    std::unordered_set<std::string> nodeStrs;
    std::unordered_set<std::string> edgeStrs;

    for(std::size_t VertexID = 0; VertexID < adjacencyList_.size(); ++VertexID) {
      const std::string FromVertexName =
          static_cast<const Derived*>(this)->getVertexNameByVertexID(VertexID);

      // Convert node to dot
      nodeStrs.emplace(std::string("node [shape = ") +
                       static_cast<const Derived*>(this)->getDotShape() + "] \"" + FromVertexName +
                       "\"");

      // Convert edge to dot
      for(const Edge& edge : *(adjacencyList_[VertexID]))
        edgeStrs.emplace(
            "\"" + FromVertexName + "\" -> \"" +
            static_cast<const Derived*>(this)->getVertexNameByVertexID(edge.ToVertexID) + "\"" +
            static_cast<const Derived*>(this)->edgeDataToDot(edge.Data));
    }

    os << "digraph G {\n"
       << "rankdir=LR;\n"
       << "size=\"8.5\" \n";
    std::copy(nodeStrs.begin(), nodeStrs.end(), std::ostream_iterator<std::string>(os, ";\n"));
    std::copy(edgeStrs.begin(), edgeStrs.end(), std::ostream_iterator<std::string>(os, ";\n"));
    os << "}\n";
  }

protected:
  std::unordered_map<int, Vertex> vertices_;
  std::vector<std::shared_ptr<EdgeList>> adjacencyList_;
};

} // namespace dawn

#endif
