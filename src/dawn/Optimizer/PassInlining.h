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

#ifndef DAWN_OPTIMIZER_PASSINLINING_H
#define DAWN_OPTIMIZER_PASSINLINING_H

#include "dawn/Optimizer/Pass.h"

namespace dawn {

/// @brief Try to inline stencil functions
///
/// Stencil functions which do not have a return are always inlined (if the pass is not disabled).
/// Depending on the strategy, stencil functions which do have a return are only inlined if we
/// favor precomputations.
///
/// If a stencil function is inlined the AST is modified and it may happen that certain statements
/// and expressions do not carry a valid SourceLocation anymore!
///
/// @ingroup optimizer
///
/// This pass is not necessary to create legal code and is hence not in the debug-group
class PassInlining : public Pass {
public:
  /// @brief Inlining strategies
  enum InlineStrategyKind {
    IK_Unknown,
    IK_None,                ///< Skip inlining pass
    IK_ComputationOnTheFly, ///< Favor computations on the fly
    IK_Precomputation       ///< Favor precomputation when possible
  };

  PassInlining(InlineStrategyKind strategy);

  /// @brief Pass implementation
  bool run(const std::shared_ptr<StencilInstantiation>& stencilInstantiation) override;

private:
  InlineStrategyKind strategy_;
};

} // namespace dawn

#endif
