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
#include "dawn/Optimizer/DependencyGraphAccesses.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/StencilInstantiation.h"

namespace dawn {

PassPrintStencilGraph::PassPrintStencilGraph() : Pass("PassPrintStencilGraph") {
  dependencies_.push_back("PassStageSplitter");
}

bool PassPrintStencilGraph::run(const std::shared_ptr<StencilInstantiation>& stencilInstantiation) {
  OptimizerContext* context = stencilInstantiation->getOptimizerContext();
  if(!context->getOptions().DumpStencilGraph)
    return true;

  int stencilIdx = 0;
  for(auto& stencilPtr : stencilInstantiation->getStencils()) {
    Stencil& stencil = *stencilPtr;
    auto DAG = std::make_shared<DependencyGraphAccesses>(stencilInstantiation.get());

    // Merge all stages into a single DAG
    int numStages = stencil.getNumStages();
    for(int i = 0; i < numStages; ++i)
      DAG->merge(stencil.getStage(i)->getSingleDoMethod().getDependencyGraph().get());

    DAG->toDot("stencil_" + stencilInstantiation->getName() + "_s" + std::to_string(stencilIdx) +
               ".dot");
    DAG->toJSON("stencil_" + stencilInstantiation->getName() + "_s" + std::to_string(stencilIdx) +
                ".json");

    stencilIdx++;
  }
  return true;
}

} // namespace dawn
