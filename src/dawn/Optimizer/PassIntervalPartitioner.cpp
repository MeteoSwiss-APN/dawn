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

#include "dawn/Optimizer/PassIntervalPartitioner.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"

namespace dawn {

bool PassIntervalPartitioner::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {

  OptimizerContext* context = stencilInstantiation->getOptimizerContext();

  if(!context->getOptions().PassTmpToFunction)
    return true;

  DAWN_ASSERT(context);

  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
    const auto& fields = stencilPtr->getFields();

    // Iterate multi-stages for the replacement of temporaries by stencil functions
    for(auto& multiStage : stencilPtr->getChildren()) {
      auto cloneMS = multiStage->clone();

      auto multiInterval = multiStage->computePartitionOfIntervals();
      for(const auto& interval : multiInterval.getIntervals()) {

        auto cloneStageIt = cloneMS->childrenBegin();
        for(auto stageIt = multiStage->childrenBegin(); stageIt != multiStage->childrenEnd();
            ++stageIt, ++cloneStageIt) {

          auto cloneDoMethodIt = (*cloneStageIt)->childrenBegin();
          while(cloneDoMethodIt != (*cloneStageIt)->childrenEnd()) {
            cloneDoMethodIt = (*cloneStageIt)->childrenErase(cloneDoMethodIt);
          }

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
