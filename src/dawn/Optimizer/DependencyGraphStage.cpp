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

#include "dawn/Optimizer/DependencyGraphStage.h"
#include "dawn/Optimizer/StencilInstantiation.h"

namespace dawn {

void DependencyGraphStage::insertEdge(int StageIDFrom, int StageIDTo) {
  Base::insertEdge(StageIDFrom, StageIDTo, DependencyGraphStage::EdgeData::EK_Depends);
}

bool DependencyGraphStage::depends(int StageIDFrom, int StageIDTo) const {
  const EdgeList& edgeList = edgesOf(StageIDFrom);

  if(edgeList.empty())
    return false;

  int VertedIDTo = getVertexIDFromID(StageIDTo);
  for(const Edge& edge : edgeList)
    if(edge.ToVertexID == VertedIDTo)
      return true;
  return false;
}

const char* DependencyGraphStage::edgeDataToString(const EdgeData& data) const {
  return " -----> ";
}

std::string DependencyGraphStage::edgeDataToDot(const EdgeData& data) const { return ""; }

std::string DependencyGraphStage::getVertexNameByVertexID(std::size_t VertexID) const {
  return stencilInstantiation_->getNameFromStageID(getIDFromVertexID(VertexID));
}

const char* DependencyGraphStage::getDotShape() const { return "box"; }

} // namespace dawn
