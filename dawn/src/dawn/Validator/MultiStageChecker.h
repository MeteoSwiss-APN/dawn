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

#include "dawn/IIR/ASTExpr.h"
#include "dawn/AST/ASTUtil.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Support/Exception.h"

namespace dawn {

//===------------------------------------------------------------------------------------------===//
//     MultiStageChecker
//===------------------------------------------------------------------------------------------===//
/// @brief Check whether multistages in stencil instantiation exceeds max halo points.
class MultiStageChecker {
public:
  MultiStageChecker();

  void run(iir::StencilInstantiation* instantiation, const int maxHaloPoints = 3);
};

} // namespace dawn
