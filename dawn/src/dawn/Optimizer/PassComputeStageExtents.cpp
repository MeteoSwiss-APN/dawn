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
#include "dawn/IIR/Extents.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Support/STLExtras.h"

namespace dawn {

PassComputeStageExtents::PassComputeStageExtents(OptimizerContext& context)
    : Pass(context, "PassComputeStageExtents", true) {}

bool PassComputeStageExtents::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  stencilInstantiation->computeStageExtents();
  return true;
}

} // namespace dawn
