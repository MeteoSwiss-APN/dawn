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

#include "dawn/Optimizer/PassComputeStageExtents.h"
#include "dawn/Optimizer/DependencyGraphStage.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/Support/STLExtras.h"

namespace dawn {

PassComputeStageExtents::PassComputeStageExtents() : Pass("PassComputeStageExtents", true) {
  dependencies_.push_back("PassSetStageName");
}

bool PassComputeStageExtents::run(
    const std::shared_ptr<StencilInstantiation>& stencilInstantiation) {
  for(auto& stencilPtr : stencilInstantiation->getStencils()) {
    Stencil& stencil = *stencilPtr;

    int numStages = stencil.getNumStages();

    // backward loop over stages
    for(int i = numStages - 1; i >= 0; --i) {
      Stage& fromStage = *(stencil.getStage(i));

      Extents const& stageExtent = fromStage.getExtents();

      // loop over all the input fields read in fromStage
      for(const Field& fromField : fromStage.getFields()) {
        // notice that IO (if read happens before write) would also be a valid pattern
        // to trigger the propagation of the stage extents, however this is not a legal
        // pattern within a stage
        if(fromField.getIntend() != Field::IntendKind::IK_Input)
          continue;

        Extents fieldExtent = fromField.getExtents();

        fieldExtent.expand(stageExtent);

        // check which (previous) stage computes the field (read in fromStage)
        for(int j = i - 1; j >= 0; --j) {
          Stage& toStage = *(stencil.getStage(j));
          auto fields = toStage.getFields();
          auto it = std::find_if(fields.begin(), fields.end(), [&](Field const& f) {
            return (f.getIntend() != Field::IntendKind::IK_Input) &&
                   (f.getAccessID() == fromField.getAccessID());
          });
          if(it == fields.end())
            continue;

          // if found, add the (read) extent of the field as an extent of the stage
          toStage.getExtents().merge(fieldExtent);
        }
      }
    }
  }

  return true;
}

} // namespace dawn
