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

#include "gsl/Optimizer/PassSetStageName.h"
#include "gsl/Optimizer/OptimizerContext.h"
#include "gsl/Optimizer/StencilInstantiation.h"

namespace gsl {

PassSetStageName::PassSetStageName() : Pass("PassSetStageName") {}

bool PassSetStageName::run(StencilInstantiation* stencilInstantiation) {
  stencilInstantiation->getStageIDToNameMap().clear();

  int stencilIdx = 0;
  for(auto& stencilPtr : stencilInstantiation->getStencils()) {
    std::string stencilName = stencilInstantiation->getName();

    if(stencilInstantiation->getStencils().size() > 1)
      stencilName += std::to_string(stencilIdx);

    int multiStageIdx = 0;
    for(auto& multiStagePtr : stencilPtr->getMultiStages()) {
      int stageIdx = 0;
      for(auto& stagePtr : multiStagePtr->getStages()) {
        Stage& stage = *stagePtr;
        stencilInstantiation->getStageIDToNameMap().emplace(
            stage.getStageID(),
            stencilName + "_ms" + std::to_string(multiStageIdx) + "_s" + std::to_string(stageIdx));
        stageIdx++;
      }
      multiStageIdx++;
    }
    stencilIdx++;
  }

  return true;
}

} // namespace gsl
