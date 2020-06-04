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

#include "dawn/Optimizer/PassIntervalPartitioning.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilInstantiation.h"

namespace dawn {

bool PassIntervalPartitioning::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const Options& options) {

  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
    for(auto& multiStage : stencilPtr->getChildren()) {
      auto cloneMS = multiStage->clone();

      auto multiInterval = multiStage->computePartitionOfIntervals();

      // empty out the multistage
      for(auto cloneStageIt = cloneMS->childrenBegin(); cloneStageIt != cloneMS->childrenEnd();
          ++cloneStageIt) {
        auto cloneDoMethodIt = (*cloneStageIt)->childrenBegin();
        while(cloneDoMethodIt != (*cloneStageIt)->childrenEnd()) {
          cloneDoMethodIt = (*cloneStageIt)->childrenErase(cloneDoMethodIt);
        }
      }

      for(const auto& interval : multiInterval.getIntervals()) {

        auto cloneStageIt = cloneMS->childrenBegin();
        for(auto stageIt = multiStage->childrenBegin(); stageIt != multiStage->childrenEnd();
            ++stageIt, ++cloneStageIt) {

          for(auto doMethodIt = (*stageIt)->childrenBegin();
              doMethodIt != (*stageIt)->childrenEnd(); ++doMethodIt) {
            if((*doMethodIt)->getInterval().overlaps(interval)) {
              auto doMAtInterval = (*doMethodIt)->clone();
              doMAtInterval->setInterval(interval);
              (*cloneStageIt)->insertChild(std::move(doMAtInterval));
            }
          }
        }
      }
      stencilPtr->replace(multiStage, cloneMS);
    }
  }

  for(auto& doMethod : iterateIIROver<iir::DoMethod>(*(stencilInstantiation->getIIR()))) {
    doMethod->update(iir::NodeUpdateType::levelAndTreeAbove);
  }

  return true;
}

} // namespace dawn
