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
#include "dawn/Optimizer/DependencyGraph.h"
#include "dawn/Optimizer/DependencyGraphAccesses.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/StringUtil.h"
#include <iostream>

namespace dawn {

PassTemporaryMerger::PassTemporaryMerger() : Pass("PassTemporaryMerger") {}

bool PassTemporaryMerger::run(const std::shared_ptr<StencilInstantiation>& stencilInstantiation) {
  OptimizerContext* context = stencilInstantiation->getOptimizerContext();

  using Edge = DependencyGraphAccesses::Edge;
  using Vertex = DependencyGraphAccesses::Vertex;

  bool merged = false;

  bool stencilNeedsMergePass = false;
  for(const auto& stencilPtr : stencilInstantiation->getStencils())
    stencilNeedsMergePass |=
        stencilPtr->getSIRStencil()->Attributes.has(sir::Attr::AK_MergeTemporaries);

  if(!(context->getOptions().MergeTemporaries || stencilNeedsMergePass))
    return true;

  // Pair of nodes to visit and AccessID of the last temporary (or -1 if no temporary has been
  // processed yet)
  std::vector<std::pair<std::size_t, int>> nodesToVisit;
  std::unordered_set<std::size_t> visitedNodes;

  int stencilIdx = 0;
  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
    Stencil& stencil = *stencilPtr;

    // Build the dependency graph of the stencil (merge all dependency graphs of the multi-stages)
    DependencyGraphAccesses AccessesDAG(stencilInstantiation.get());
    for(const auto& multiStagePtr : stencilPtr->getMultiStages()) {
      MultiStage& multiStage = *multiStagePtr;
      AccessesDAG.merge(multiStage.getDependencyGraphOfAxis().get());
    }
    const auto& adjacencyList = AccessesDAG.getAdjacencyList();

    // Build the dependency graph of the temporaries
    DependencyGraphAccesses TemporaryDAG(stencilInstantiation.get());
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

        if(stencilInstantiation->isTemporaryField(FromAccessID)) {
          TemporaryDAG.insertNode(FromAccessID);
          AccessIDOfLastTemporary = FromAccessID;
        }

        // Check if we already visited this node
        if(visitedNodes.count(FromVertexID))
          continue;
        else
          visitedNodes.insert(FromVertexID);

        // Follow edges of the current node and update the node extents
        for(const Edge& edge : *adjacencyList[FromVertexID]) {
          std::size_t ToVertexID = edge.ToVertexID;
          int ToAccessID = AccessesDAG.getIDFromVertexID(ToVertexID);
          int newAccessIDOfLastTemporary = AccessIDOfLastTemporary;

          if(stencilInstantiation->isTemporaryField(ToAccessID) && AccessIDOfLastTemporary != -1) {
            TemporaryDAG.insertEdge(AccessIDOfLastTemporary, ToAccessID);
            newAccessIDOfLastTemporary = ToAccessID;
          }

          nodesToVisit.emplace_back(ToVertexID, newAccessIDOfLastTemporary);
        }
      }
    }

    std::cout << TemporaryDAG.toDot() << std::endl;

    if(TemporaryDAG.empty())
      continue;

    // Add dependencies due to overlapping lifetime of the temporaries. The dependencies will assure
    // that these temporaries can not be merged into one i.e have different colors.
    std::unordered_set<int> temporaries;
    std::for_each(TemporaryDAG.getVertices().begin(), TemporaryDAG.getVertices().end(),
                  [&](const std::pair<int, Vertex>& vertexPair) {
                    temporaries.emplace(vertexPair.second.ID);
                  });
    auto LifeTimeMap = stencil.getLifetime(temporaries);

    for(const auto& FromAccessIDLifetimePair : LifeTimeMap) {
      const int FromAccessID = FromAccessIDLifetimePair.first;
      const Stencil::Lifetime& FromLifetime = FromAccessIDLifetimePair.second;

      for(const auto& ToAccessIDLifetimePair : LifeTimeMap) {
        const int ToAccessID = ToAccessIDLifetimePair.first;
        const Stencil::Lifetime& ToLifetime = ToAccessIDLifetimePair.second;

        if(FromAccessID == ToAccessID)
          continue;

        if(FromLifetime.overlaps(ToLifetime)) {
          TemporaryDAG.insertEdge(FromAccessID, ToAccessID);
        }
      }
    }

    if(context->getOptions().DumpTemporaryGraphs)
      TemporaryDAG.toDot(format("tmp_stencil_%i.dot", stencilIdx));

    // Color the temporary graph
    std::unordered_map<int, int> coloring;
    TemporaryDAG.greedyColoring(coloring);

    // Compute the candiates for renaming. We will rename all fields of the same color to the same
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
      const std::vector<int>& AccessIDOfRenameCandiates = colorRenameCandidatesPair.second;

      // Print the rename candiates in alphabetical order
      if(context->getOptions().ReportPassTemporaryMerger && AccessIDOfRenameCandiates.size() >= 2) {
        std::vector<std::string> renameCandiatesNames;
        for(int AccessID : AccessIDOfRenameCandiates)
          renameCandiatesNames.emplace_back(stencilInstantiation->getNameFromAccessID(AccessID));
        std::sort(renameCandiatesNames.begin(), renameCandiatesNames.end());
        std::cout << "\nPASS: " << getName() << ": " << stencilInstantiation->getName()
                  << ": merging: " << RangeToString(", ", "", "\n")(renameCandiatesNames);
      }

      int newAccessID = AccessIDOfRenameCandiates[0];

      // Rename all other fields of the color to the AccessID of the first field (note that it
      // wouldn't matter which AccessID we choose)
      for(int i = 1; i < AccessIDOfRenameCandiates.size(); ++i) {
        merged = true;
        int oldAccessID = AccessIDOfRenameCandiates[i];
        stencilInstantiation->renameAllOccurrences(stencilPtr.get(), oldAccessID, newAccessID);
      }
    }

    stencilIdx++;
  }

  if(context->getOptions().ReportPassTemporaryMerger && !merged)
    std::cout << "\nPASS: " << getName() << ": " << stencilInstantiation->getName()
              << ": no merge\n";

  return true;
}

} // namespace dawn
