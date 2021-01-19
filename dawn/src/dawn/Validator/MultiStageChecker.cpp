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
#include "dawn/Validator/MultiStageChecker.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Support/Exception.h"

namespace dawn {
MultiStageChecker::MultiStageChecker() {}

void MultiStageChecker::run(iir::StencilInstantiation* instantiation,
                            const int maxHaloPoints) {
  iir::Extents maxExtents{ast::cartesian};
  for(const auto& stencil : instantiation->getStencils()) {
    // Merge stencil field extents...
    for(const auto& fieldPair : stencil->getOrderedFields()) {
      const auto& fieldInfo = fieldPair.second;
      if(!fieldInfo.IsTemporary) {
        const auto& fieldExtents = fieldInfo.field.getExtentsRB();
        maxExtents.merge(fieldExtents);
      }
    }
    // Merge stencil stage extents...
    for(const auto& multistage : stencil->getChildren()) {
      for(const auto& stage : multistage->getChildren()) {
        const auto& stageExtents = stage->getExtents();
        maxExtents.merge(stageExtents);
      }
    }
  }
  // Check if max extents exceed max halo points...
  const auto& horizExtent =
      iir::extent_cast<iir::CartesianExtent const&>(maxExtents.horizontalExtent());
  const auto& vertExtent = maxExtents.verticalExtent();
  if(horizExtent.iPlus() > maxHaloPoints || horizExtent.iMinus() < -maxHaloPoints ||
     horizExtent.jPlus() > maxHaloPoints || horizExtent.jMinus() < -maxHaloPoints ||
     vertExtent.plus() > maxHaloPoints || vertExtent.minus() < -maxHaloPoints) {
    throw CompileError("Multistage exeeds max halo points " + std::to_string(maxHaloPoints));
  }
}

} // namespace dawn
