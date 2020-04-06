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
    : instantiation_(instantiation), metadata_(instantiation->getMetaData()),
      maxHaloPoints_(maxHaloPoints) {}

void MultiStageChecker::run() {
  for(const auto& stencil : instantiation_->getStencils()) {
    for(const auto& multistage : stencil->getChildren()) {
      for(const auto& stage : multistage->getChildren()) {
        for(const auto& doMethod : stage->getChildren()) {
          for(const auto& accessPair : metadata_.getAccessIDToNameMap()) {
            std::cerr << "MulitStageChecker: check field '" << accessPair.second << "'\n";
            auto maxExtent = doMethod->computeMaximumExtents(accessPair.first);
            if(maxExtent) {
              const auto& horizExtent = iir::extent_cast<iir::CartesianExtent const&>(
                  maxExtent.value().horizontalExtent());
              const auto& vertExtent = maxExtent.value().verticalExtent();
              if(horizExtent.iPlus() > maxHaloPoints_ || horizExtent.iMinus() < -maxHaloPoints_ ||
                 horizExtent.jPlus() > maxHaloPoints_ || horizExtent.jMinus() < -maxHaloPoints_ ||
                 vertExtent.plus() > maxHaloPoints_ || vertExtent.minus() < -maxHaloPoints_) {
                throw CompileError("Multistage exeeds max halo points " +
                                   std::to_string(maxHaloPoints_) + " in field access '" +
                                   accessPair.second + "'");
              }
            }
          }
        }
      }
    }
  }
}

} // namespace dawn
