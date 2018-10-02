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

#include "dawn/IIR/DependencyGraphStage.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/Renaming.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/StringUtil.h"
#include "dawn/Support/Unreachable.h"
#include <algorithm>
#include <iostream>
#include <numeric>

namespace dawn {
namespace iir {

IIR::IIR() : metadata_(std::make_shared<StencilMetaInformation>()) {}

std::unique_ptr<IIR> IIR::clone() const {

  auto cloneIIR = make_unique<IIR>();

  cloneIIR->cloneChildrenFrom(*this, cloneIIR);
  return cloneIIR;
}

Options& IIR::getOptions() { return creator_->getOptions(); }

const DiagnosticsEngine& IIR::getDiagnostics() const { return creator_->getDiagnostics(); }
DiagnosticsEngine& IIR::getDiagnostics() { return creator_->getDiagnostics(); }

const HardwareConfig& IIR::getHardwareConfiguration() const {
  return creator_->getHardwareConfiguration();
}

HardwareConfig& IIR::getHardwareConfiguration() { return creator_->getHardwareConfiguration(); }

} // namespace iir
} // namespace dawn
