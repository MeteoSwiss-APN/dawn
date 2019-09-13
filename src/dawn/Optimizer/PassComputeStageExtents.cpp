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
#include "dawn/IIR/DependencyGraphStage.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Support/STLExtras.h"

namespace dawn {

PassComputeStageExtents::PassComputeStageExtents(OptimizerContext& context)
    : Pass(context, "PassComputeStageExtents", true) {
  dependencies_.push_back("PassSetStageName");
}

bool PassComputeStageExtents::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
    iir::Stencil& stencil = *stencilPtr;

    int numStages = stencil.getNumStages();

    // backward loop over stages
    for(int i = numStages - 1; i >= 0; --i) {
      iir::Stage& fromStage = *(stencil.getStage(i));

      iir::Extents const& stageExtent = fromStage.getExtents();

      // loop over all the input fields read in fromStage
      for(const auto& fromFieldPair : fromStage.getFields()) {

        const iir::Field& fromField = fromFieldPair.second;
        auto&& fromFieldExtents = fromField.getExtents();

        // notice that IO (if read happens before write) would also be a valid pattern
        // to trigger the propagation of the stage extents, however this is not a legal
        // pattern within a stage
        // ===-----------------------------------------------------------------------------------===
        //      Point one [ExtentComputationTODO]
        // ===-----------------------------------------------------------------------------------===

        iir::Extents fieldExtent = fromFieldExtents;

        fieldExtent.expand(stageExtent);

        // check which (previous) stage computes the field (read in fromStage)
        for(int j = i - 1; j >= 0; --j) {
          iir::Stage& toStage = *(stencil.getStage(j));
          // ===---------------------------------------------------------------------------------===
          //      Point two [ExtentComputationTODO]
          // ===---------------------------------------------------------------------------------===
          auto fields = toStage.getFields();
          auto it = std::find_if(fields.begin(), fields.end(),
                                 [&](std::pair<int, iir::Field> const& pair) {
                                   const auto& f = pair.second;
                                   return (f.getIntend() != iir::Field::IntendKind::IK_Input) &&
                                          (f.getAccessID() == fromField.getAccessID());
                                 });
          if(it == fields.end())
            continue;

          // if found, add the (read) extent of the field as an extent of the stage
          iir::Extents ext = toStage.getExtents();
          ext.merge(fieldExtent);
          // this pass is computing the redundant computation in the horizontal, therefore we
          // nullify the vertical component of the stage
          ext[2] = iir::Extent{0, 0};
          toStage.setExtents(ext);
        }
      }
    }
  }

  for(const auto& MS : iterateIIROver<iir::MultiStage>(*(stencilInstantiation->getIIR()))) {
    MS->update(iir::NodeUpdateType::levelAndTreeAbove);
  }

  return true;
}

} // namespace dawn
