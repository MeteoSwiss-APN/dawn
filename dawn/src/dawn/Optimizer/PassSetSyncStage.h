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

#include "dawn/IIR/Cache.h"
#include "dawn/IIR/Interval.h"
#include "dawn/IIR/Stage.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/Optimizer/Pass.h"

namespace dawn {

/// @brief Determine whether stages require a synchronization
/// input: well formed and legal IIR
/// output: IIR with sync property of derived info of stage
/// @ingroup optimizer
///
class PassSetSyncStage : public Pass {
public:
  PassSetSyncStage(OptimizerContext& context);

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) override;

private:
  bool requiresSync(const iir::Stage& stage, const std::unique_ptr<iir::MultiStage>& ms) const;
};

} // namespace dawn
