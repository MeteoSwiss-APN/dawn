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

#ifndef GENERATE_IN_MEMORY_STENCILS_H
#define GENERATE_IN_MEMORY_STENCILS_H

#include "dawn/IIR/IIR.h"
#include "dawn/IIR/StencilInstantiation.h"
#include <memory>

std::shared_ptr<dawn::iir::StencilInstantiation>
createCopyStencilIIRInMemory(dawn::ast::GridType gridType);

std::shared_ptr<dawn::iir::StencilInstantiation>
createLapStencilIIRInMemory(dawn::ast::GridType gridType);

std::shared_ptr<dawn::iir::StencilInstantiation> createUnstructuredSumEdgeToCellsIIRInMemory();

std::shared_ptr<dawn::iir::StencilInstantiation> createUnstructuredMixedCopies();

#endif
