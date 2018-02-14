#include "dawn/Optimizer/StronglyConnectedComponents.h"
#include "dawn/Optimizer/DependencyGraphAccesses.h"

namespace dawn {

// using VertexType = DependencyGraphAccesses::Vertex;
using EdgeType = DependencyGraphAccesses::Edge;

bool StronglyConnectedComponents::find() {
  if(scc_)
    scc_->clear();

  // Initialize the vertex Data
  for(const auto& AccessIDVertexPair : graph_->getVertices()) {
    const auto& vertex = AccessIDVertexPair.second;
    vertexData_.emplace(vertex.VertexID, VertexData{-1, -1, false, vertex.ID});
  }
  index_ = 0;
  hasMultiNodeSCC_ = false;

  // Iterate all Vertices
  for(const auto& AccessIDVertexPair : graph_->getVertices()) {
    const auto& vertex = AccessIDVertexPair.second;
    findImpl(vertex.VertexID);
  }

  return hasMultiNodeSCC_;
}

void StronglyConnectedComponents::findImpl(std::size_t FromVertexID) {
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
  for(const EdgeType& edge : *graph_->getAdjacencyList()[FromVertexID]) {

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
      SCC.insert(vertexData.ID);

    } while(VertexID != FromVertexID);

    if(SCC.size() > 1) {
      hasMultiNodeSCC_ = true;
      if(scc_)
        scc_->emplace_back(std::move(SCC));
    }
  }
}

} // namespace dawn
