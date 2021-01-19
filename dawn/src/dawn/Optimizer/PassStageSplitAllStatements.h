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

#include "dawn/Optimizer/Pass.h"

namespace dawn {

/// @brief
/// * Input: Any IIR
/// * Output: same IIR, but with stages split on each statement
/// @ingroup optimizer
///
/// This pass is necessary to generate legal IIR for GridType == Unstructured.
///
class PassStageSplitAllStatements : public Pass {
public:
  PassStageSplitAllStatements() : Pass("PassStageSplitAllStatements") {}

  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
           const Options& options = {}) override;
};

} // namespace dawn
