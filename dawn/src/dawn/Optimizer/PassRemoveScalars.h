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

#include "Pass.h"

namespace dawn {

/// @brief PassRemoveScalars will remove local variables flagged as scalars from each DoMethod in
/// the IIR tree.
/// * Input:  IIR with local variables' types (iir::LocalVariableData::type_) correctly computed by
///           PassLocalVariableType.
/// * Output: same as input, but with scalar local variables inlined.
/// @ingroup optimizer
///
/// This pass is necessary to generate legal IIR
class PassRemoveScalars : public Pass {
public:
  PassRemoveScalars(OptimizerContext& context) : Pass(context, "PassRemoveScalars") {}

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) override;
};

} // namespace dawn
