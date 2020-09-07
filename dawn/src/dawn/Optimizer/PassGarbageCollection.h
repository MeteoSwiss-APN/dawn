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
/// * Input: any IIR
/// * Output: same IIR, but empy IIR nodes are pruned. IIR nodes can become empty due to passes like
////          PassRemoveScalars.
/// @ingroup optimizer
///
/// This pass is necessary to generate legal IIR since empty statement lists can cause some passes
/// to fail
class PassGarbageCollection : public Pass {
public:
  PassGarbageCollection() : Pass("PassGarbageCollection") {}

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
           const Options& options = {}) override;
};

} // namespace dawn
