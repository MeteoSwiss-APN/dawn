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
  if(!context_.getOptions().PartitionIntervals) {
    return true;
  }

  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
    for(const auto& multiStage : stencilPtr->getChildren()) {

      // Compute the partition of intervals. Assumption is that the returned partition is order
      // according to an ascending order of intervals.
      auto multiInterval = multiStage->computePartitionOfIntervals();

      for(const auto& interval : multiInterval.getIntervals()) {
        for(auto& stage : multiStage->getChildren()) {
          for(auto doMethodIt = stage->childrenBegin(); doMethodIt != stage->childrenEnd();
              ++doMethodIt) {
            auto& doMethod = *doMethodIt;
            if(doMethod->getInterval().overlaps(interval)) {
              // Create a clone of the current doMethod, set its interval to `interval' and insert
              // it into the current stage before the current doMethod.
              auto doMAtInterval = doMethod->clone();
              doMAtInterval->setInterval(interval);
              doMethodIt = stage->insertChild(doMethodIt, std::move(doMAtInterval));
              ++doMethodIt; // this should point back to the original doMethod
              // The current doMethod shouldn't cover the interval that is now covered by the new
              // clone. So we carve it out.
              doMethod->setInterval(doMethod->getInterval().carve(interval));
            }
          }
        }
      }
    }
  }

  for(auto& doMethod : iterateIIROver<iir::DoMethod>(*(stencilInstantiation->getIIR()))) {
    doMethod->update(iir::NodeUpdateType::levelAndTreeAbove);
  }

  return true;
}

} // namespace dawn
