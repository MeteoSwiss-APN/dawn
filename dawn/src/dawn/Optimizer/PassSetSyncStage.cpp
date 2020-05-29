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

#include "dawn/Optimizer/PassSetSyncStage.h"
#include "dawn/IIR/Cache.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/IntervalAlgorithms.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Support/Unreachable.h"

#include <optional>
#include <set>
#include <vector>

namespace dawn {

bool PassSetSyncStage::requiresSync(const iir::Stage& stage,
                                    const std::unique_ptr<iir::MultiStage>& ms) const {
  const int stageId = stage.getStageID();

  // if there is only one stage, there can not be horizontal data dependencies
  if(ms->getChildren().size() <= 1) {
    return false;
  }
  DAWN_ASSERT(!ms->getChildren().empty());

  for(const auto& field : stage.getFields()) {
    const int accessID = field.second.getAccessID();

    // if the intend of the field is output, we wont require sync from this stage
    // if the intend were IO, it might still not be needed if the output happens before a read,
    // however to simplify the algorithm we will be conservative and apply a sync here
    if(field.second.getIntend() == iir::Field::IntendKind::Output) {
      continue;
    }

    // we dont consider the field if the access is pointwise
    if(field.second.getExtents().isHorizontalPointwise())
      continue;

    std::optional<iir::Interval> fieldInterval =
        stage.computeEnclosingAccessInterval(accessID, false);
    DAWN_ASSERT(fieldInterval);

    for(const auto& stageIt : ms->getChildren()) {
      if(stageIt->getStageID() == stageId)
        break;

      // we only process those fields where the stage compute
      const auto& preFields = stageIt->getFields();
      if(!preFields.count(accessID) ||
         (preFields.at(accessID).getIntend() == iir::Field::IntendKind::Input))
        continue;

      if(preFields.at(accessID).computeAccessedInterval().overlaps(*fieldInterval))
        return true;
    }
  }
  return false;
}

PassSetSyncStage::PassSetSyncStage(OptimizerContext& context) : Pass(context, "PassSetSyncStage") {}

bool PassSetSyncStage::run(const std::shared_ptr<iir::StencilInstantiation>& instantiation) {
  // Update derived info
  instantiation->computeDerivedInfo();

  for(const auto& ms : iterateIIROver<iir::MultiStage>(*(instantiation->getIIR()))) {
    for(const auto& stage : ms->getChildren()) {
      // the last stage also requires a sync, since it will be written into by the next k level
      // processing
      if(requiresSync(*stage, ms)) {
        stage->setRequiresSync(true);
      } else {
        stage->setRequiresSync(false);
      }
    }
  }
  return true;
}

} // namespace dawn
