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

#ifndef DAWN_OPTIMIZER_MULTISTAGECHECKER_H
#define DAWN_OPTIMIZER_MULTISTAGECHECKER_H

#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTUtil.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Support/Exception.h"

namespace dawn {

//===------------------------------------------------------------------------------------------===//
//     MultiStageChecker
//===------------------------------------------------------------------------------------------===//
/// @brief Perform basic integrity checks on the AST.
class MultiStageChecker {
  iir::StencilInstantiation* instantiation_;
  iir::StencilMetaInformation& metadata_;
  int maxHaloPoints_;

public:
  MultiStageChecker(iir::StencilInstantiation* instantiation, const int maxHaloPoints = 3);

  void run();
};

} // namespace dawn

#endif // DAWN_OPTIMIZER_MULTISTAGECHECKER_H
