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
MultiStageChecker::MultiStageChecker(iir::StencilInstantiation* instantiation,
                                     const int maxHaloPoints)
    : instantiation_(instantiation), maxHaloPoints_(maxHaloPoints) {}

void MultiStageChecker::run() {
  iir::Extents maxExtents{ast::cartesian};
  for(const auto& stencil : instantiation_->getStencils()) {
    // Merge stencil field extents...
    for(const auto& fieldPair : stencil->getOrderedFields()) {
      const auto& fieldInfo = fieldPair.second;
      if(!fieldInfo.IsTemporary) {
        const auto& fieldExtents = fieldInfo.field.getExtentsRB();
        maxExtents.merge(fieldExtents);
      }
    }
  }

  // Get horizontal extents
  int iMinus, iPlus, jMinus, jPlus;
  try {
    const auto& horizExtent =
        iir::extent_cast<iir::CartesianExtent const&>(maxExtents.horizontalExtent());
    iMinus = horizExtent.iMinus();
    iPlus = horizExtent.iPlus();
    jMinus = horizExtent.jMinus();
    jPlus = horizExtent.jPlus();
  } catch(const std::bad_cast& error) {
    iMinus = iPlus = jMinus = jPlus = 0;
  }

  // Get vertical extents
  const auto& vertExtent = maxExtents.verticalExtent();
  int kMinus = vertExtent.minus();
  int kPlus = vertExtent.plus();

  // Check if max extents exceed max halo points...
  if(iPlus > maxHaloPoints_ || iMinus < -maxHaloPoints_ || jPlus > maxHaloPoints_ ||
     jMinus < -maxHaloPoints_ || kPlus > maxHaloPoints_ || kMinus < -maxHaloPoints_) {
    throw CompileError(
        "Multistage extent (" + std::to_string(iMinus) + "," + std::to_string(iPlus) + "," +
        std::to_string(jMinus) + "," + std::to_string(jPlus) + "," + std::to_string(kMinus) + "," +
        std::to_string(kPlus) + ") exeeds max halo points " + std::to_string(maxHaloPoints_));
  }
}

} // namespace dawn
