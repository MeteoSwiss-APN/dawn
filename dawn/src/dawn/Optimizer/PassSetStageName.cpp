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

#include "dawn/Optimizer/PassSetStageName.h"
#include "dawn/IIR/StencilInstantiation.h"

namespace dawn {

bool PassSetStageName::run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                           const Options& options) {
  stencilInstantiation->getIIR()->getStageIDToNameMap().clear();

  int stencilIdx = 0;
  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
    std::string stencilName = stencilInstantiation->getName();

    if(stencilInstantiation->getStencils().size() > 1)
      stencilName += std::to_string(stencilIdx);

    int multiStageIdx = 0;
    for(const auto& multiStagePtr : stencilPtr->getChildren()) {
      int stageIdx = 0;
      for(const auto& stagePtr : multiStagePtr->getChildren()) {
        iir::Stage& stage = *stagePtr;
        stencilInstantiation->getIIR()->getStageIDToNameMap().emplace(
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

} // namespace dawn
