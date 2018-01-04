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

PassComputeStageExtents::PassComputeStageExtents() : Pass("PassComputeStageExtents") {
  dependencies_.push_back("PassSetStageName");
}

bool PassComputeStageExtents::run(StencilInstantiation* stencilInstantiation) {
  for(auto& stencilPtr : stencilInstantiation->getStencils()) {
    Stencil& stencil = *stencilPtr;

    int numStages = stencil.getNumStages();

    for(int i = numStages - 1; i >= 0; --i) {
      Stage& fromStage = *(stencil.getStage(i));

      Extents const& stageExtent = fromStage.getExtents();

      for(const Field& fromField : fromStage.getFields()) {
        // notice that IO (if read happens before write) would also be a valid pattern
        // to trigger the propagation of the stage extents, however this is not a legal
        // pattern within a stage if the extent is not pointwise
        if(fromField.Extent.isPointwise() || fromField.Intend != Field::IntendKind::IK_Input)
          continue;

        Extents fieldExtent = fromField.Extent;

        fieldExtent.expand(stageExtent);

        for(int j = i - 1; j >= 0; --j) {
          Stage& toStage = *(stencil.getStage(j));
          auto fields = toStage.getFields();
          auto it = std::find_if(fields.begin(), fields.end(), [&](Field const& f) {
            return (f.Intend != Field::IntendKind::IK_Input) && (f.AccessID == fromField.AccessID);
          });
          if(it == fields.end())
            continue;

          toStage.getExtents().merge(fieldExtent);
        }
      }
    }
  }

  return true;
}

} // namespace dawn
