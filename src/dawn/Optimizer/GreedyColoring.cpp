#include "dawn/Optimizer/GreedyColoring.h"
#include "dawn/Optimizer/DependencyGraphAccesses.h"

namespace dawn {
void GreedyColoring::compute() {
  using Edge = DependencyGraphAccesses::Edge;

  coloring_.clear();

  std::size_t numVertices = graph_->getNumVertices();
  const auto& adjacencyList = graph_->getAdjacencyList();

  // Compute the neighbor-list
  std::vector<std::set<std::size_t>> neighborList(numVertices);
  for(std::size_t FromVertexID = 0; FromVertexID < numVertices; ++FromVertexID) {
    for(const Edge& edge : *adjacencyList[FromVertexID]) {
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

} // namespace dawn
