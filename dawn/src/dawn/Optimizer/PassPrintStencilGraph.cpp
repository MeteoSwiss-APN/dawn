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

#include "dawn/Optimizer/PassPrintStencilGraph.h"
#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/IIR/StencilInstantiation.h"

namespace dawn {

bool PassPrintStencilGraph::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const Options& options) {

  int stencilIdx = 0;
  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
    iir::Stencil& stencil = *stencilPtr;
    iir::DependencyGraphAccesses DAG(stencilInstantiation->getMetaData());

    // Merge all stages into a single DAG
    int numStages = stencil.getNumStages();
    for(int i = 0; i < numStages; ++i)
      DAG.merge(*stencil.getStage(i)->getSingleDoMethod().getDependencyGraph());

    DAG.toDot("stencil_" + stencilInstantiation->getName() + "_s" + std::to_string(stencilIdx) +
              ".dot");
    DAG.toJSON("stencil_" + stencilInstantiation->getName() + "_s" + std::to_string(stencilIdx) +
               ".json");

    stencilIdx++;
  }
  return true;
}

} // namespace dawn
