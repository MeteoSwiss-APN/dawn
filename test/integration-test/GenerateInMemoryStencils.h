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

#pragma once

#include "dawn/IIR/IIR.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"

#include <memory>

std::shared_ptr<dawn::iir::StencilInstantiation>
createCopyStencilIIRInMemory(dawn::OptimizerContext& optimizer);
std::shared_ptr<dawn::iir::StencilInstantiation>
createLapStencilIIRInMemory(dawn::OptimizerContext& optimizer);