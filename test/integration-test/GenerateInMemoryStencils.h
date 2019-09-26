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

#include <memory>

#include "dawn/IIR/IIR.h"
#include "dawn/IIR/StencilInstantiation.h"

using namespace dawn;

void createCopyStencilIIRInMemory(std::shared_ptr<iir::StencilInstantiation>& target);
void createLapStencilIIRInMemory(std::shared_ptr<iir::StencilInstantiation>& target);