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
#include "dawn/Optimizer/OptimizerContext.h"

namespace dawn {

PassPrintStencilGraph::PassPrintStencilGraph() : Pass("PassPrintStencilGraph") {
  dependencies_.push_back("PassStageSplitter");
}

bool PassPrintStencilGraph::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  if(!stencilInstantiation->getIIR()->getOptions().DumpStencilGraph)
    return true;

  int stencilIdx = 0;
  for(const auto& stencilPtr : stencilInstantiation->getIIR()->getChildren()) {
    iir::Stencil& stencil = *stencilPtr;
    auto DAG = std::make_shared<iir::DependencyGraphAccesses>(stencilInstantiation->getIIR().get());

    // Merge all stages into a single DAG
    int numStages = stencil.getNumStages();
    for(int i = 0; i < numStages; ++i)
      DAG->merge(stencil.getStage(i)->getSingleDoMethod().getDependencyGraph().get());

    DAG->toDot("stencil_" + stencilInstantiation->getIIR()->getMetaData()->getName() + "_s" +
               std::to_string(stencilIdx) + ".dot");
    DAG->toJSON("stencil_" + stencilInstantiation->getIIR()->getMetaData()->getName() + "_s" +
                std::to_string(stencilIdx) + ".json");

    stencilIdx++;
  }
  return true;
}

} // namespace dawn
