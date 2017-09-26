//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gsl/Optimizer/PassPrintStencilGraph.h"
#include "gsl/Optimizer/DependencyGraphAccesses.h"
#include "gsl/Optimizer/OptimizerContext.h"
#include "gsl/Optimizer/StencilInstantiation.h"

namespace gsl {

PassPrintStencilGraph::PassPrintStencilGraph() : Pass("PassPrintStencilGraph") {
  dependencies_.push_back("PassStageSplitter");
}

bool PassPrintStencilGraph::run(StencilInstantiation* stencilInstantiation) {
  OptimizerContext* context = stencilInstantiation->getOptimizerContext();
  if(context->getOptions().DumpStencilGraph) {

    int stencilIdx = 0;
    for(auto& stencilPtr : stencilInstantiation->getStencils()) {
      Stencil& stencil = *stencilPtr;
      auto DAG = std::make_shared<DependencyGraphAccesses>(stencilInstantiation);

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
  }

  return true;
}

} // namespace gsl
