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

#include "dawn/Optimizer/PassTemporaryMerger.h"
#include "dawn/IIR/DependencyGraph.h"
#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/Renaming.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/Logger.h"
#include "dawn/Support/StringUtil.h"

namespace dawn {

PassTemporaryMerger::PassTemporaryMerger(OptimizerContext& context)
    : Pass(context, "PassTemporaryMerger") {}

bool PassTemporaryMerger::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  using Edge = iir::DependencyGraphAccesses::Edge;
  using Vertex = iir::DependencyGraphAccesses::Vertex;
  const iir::StencilMetaInformation& metadata = stencilInstantiation->getMetaData();

  bool merged = false;

  // Pair of nodes to visit and AccessID of the last temporary (or -1 if no temporary has been
  // processed yet)
  std::vector<std::pair<std::size_t, int>> nodesToVisit;
  std::unordered_set<std::size_t> visitedNodes;

  int stencilIdx = 0;
  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
    iir::Stencil& stencil = *stencilPtr;

    // Build the dependency graph of the stencil (merge all dependency graphs of the multi-stages)
    iir::DependencyGraphAccesses AccessesDAG(stencilInstantiation->getMetaData());
    for(const auto& multiStagePtr : stencilPtr->getChildren()) {
      iir::MultiStage& multiStage = *multiStagePtr;
      AccessesDAG.merge(multiStage.getDependencyGraphOfAxis());
    }
    const auto& adjacencyList = AccessesDAG.getAdjacencyList();

    // Build the dependency graph of the temporaries
    iir::DependencyGraphAccesses TemporaryDAG(stencilInstantiation->getMetaData());
    int AccessIDOfLastTemporary = -1;
    for(std::size_t VertexID : AccessesDAG.getOutputVertexIDs()) {
      nodesToVisit.clear();
      visitedNodes.clear();

      nodesToVisit.emplace_back(VertexID, AccessIDOfLastTemporary = -1);
      while(!nodesToVisit.empty()) {

        // Process the current node
        std::size_t FromVertexID = nodesToVisit.back().first;
        int FromAccessID = AccessesDAG.getIDFromVertexID(FromVertexID);
        AccessIDOfLastTemporary = nodesToVisit.back().second;
        nodesToVisit.pop_back();

        if(metadata.isAccessType(iir::FieldAccessType::StencilTemporary, FromAccessID)) {
          TemporaryDAG.insertNode(FromAccessID);
          AccessIDOfLastTemporary = FromAccessID;
        }

        // Check if we already visited this node
        if(visitedNodes.count(FromVertexID))
          continue;
        else
          visitedNodes.insert(FromVertexID);

        // Follow edges of the current node and update the node extents
        for(const Edge& edge : adjacencyList[FromVertexID]) {
          std::size_t ToVertexID = edge.ToVertexID;
          int ToAccessID = AccessesDAG.getIDFromVertexID(ToVertexID);
          int newAccessIDOfLastTemporary = AccessIDOfLastTemporary;

          if(metadata.isAccessType(iir::FieldAccessType::StencilTemporary, ToAccessID) &&
             AccessIDOfLastTemporary != -1) {
            TemporaryDAG.insertEdge(AccessIDOfLastTemporary, ToAccessID, iir::Extents{});
            newAccessIDOfLastTemporary = ToAccessID;
          }

          nodesToVisit.emplace_back(ToVertexID, newAccessIDOfLastTemporary);
        }
      }
    }

    // TODO Should this be an optional dump?
    DAWN_LOG(INFO) << TemporaryDAG.toDot();

    if(TemporaryDAG.empty())
      continue;

    // Add dependencies due to overlapping lifetime of the temporaries. The dependencies will assure
    // that these temporaries can not be merged into one i.e have different colors.
    std::unordered_set<int> temporaries;
    std::for_each(TemporaryDAG.getVertices().begin(), TemporaryDAG.getVertices().end(),
                  [&](const std::pair<int, Vertex>& vertexPair) {
                    temporaries.emplace(vertexPair.second.Value);
                  });
    auto LifeTimeMap = stencil.getLifetime(temporaries);

    for(const auto& FromAccessIDLifetimePair : LifeTimeMap) {
      const int FromAccessID = FromAccessIDLifetimePair.first;
      const iir::Stencil::Lifetime& FromLifetime = FromAccessIDLifetimePair.second;

      for(const auto& ToAccessIDLifetimePair : LifeTimeMap) {
        const int ToAccessID = ToAccessIDLifetimePair.first;
        const iir::Stencil::Lifetime& ToLifetime = ToAccessIDLifetimePair.second;

        if(FromAccessID == ToAccessID)
          continue;

        if(FromLifetime.overlaps(ToLifetime)) {
          TemporaryDAG.insertEdge(FromAccessID, ToAccessID, iir::Extents{});
        }
      }
    }

    if(context_.getOptions().DumpTemporaryGraphs)
      TemporaryDAG.toDot(format("tmp_stencil_%i.dot", stencilIdx));

    // Color the temporary graph
    std::unordered_map<int, int> coloring;
    TemporaryDAG.greedyColoring(coloring);

    // Compute the candidates for renaming. We will rename all fields of the same color to the same
    // AccessID.
    std::unordered_map<int, std::vector<int>> colorToAccessIDOfRenameCandidatesMap;
    for(const auto& colorAccessIDPair : coloring) {
      int AccessID = colorAccessIDPair.first;
      int color = colorAccessIDPair.second;
      auto it = colorToAccessIDOfRenameCandidatesMap.find(color);
      if(it == colorToAccessIDOfRenameCandidatesMap.end())
        colorToAccessIDOfRenameCandidatesMap.emplace(color, std::vector<int>{AccessID});
      else
        colorToAccessIDOfRenameCandidatesMap[color].push_back(AccessID);
    }

    for(const auto& colorRenameCandidatesPair : colorToAccessIDOfRenameCandidatesMap) {
      const std::vector<int>& AccessIDOfRenameCandidates = colorRenameCandidatesPair.second;

      // Sort the rename candidates in alphabetical order
      std::vector<std::string> renameCandidatesNames;
      for(int AccessID : AccessIDOfRenameCandidates)
        renameCandidatesNames.emplace_back(metadata.getFieldNameFromAccessID(AccessID));
      std::sort(renameCandidatesNames.begin(), renameCandidatesNames.end());

      // Print the rename candidates in alphabetical order
      if(AccessIDOfRenameCandidates.size() >= 2) {
        DAWN_LOG(INFO) << stencilInstantiation->getName()
                       << ": merging: " << RangeToString(", ", "", "\n")(renameCandidatesNames);
      }

      // Rename all other fields of the color to the AccessID of the first field in alphabetical
      // order (note that it wouldn't matter which AccessID we choose).
      int newAccessID = metadata.getAccessIDFromName(renameCandidatesNames[0]);
      for(int i = 0; i < AccessIDOfRenameCandidates.size(); ++i) {
        int oldAccessID = AccessIDOfRenameCandidates[i];
        if(oldAccessID != newAccessID) {
          merged = true;
          renameAccessIDInStencil(stencilPtr.get(), oldAccessID, newAccessID);
        }
      }
    }

    stencilIdx++;
  }

  if(!merged)
    DAWN_LOG(INFO) << stencilInstantiation->getName() << ": No merge";

  return true;
}

} // namespace dawn
