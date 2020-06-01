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
#include <unordered_map>

namespace dawn {

/// @brief This pass checks the code and inserts halo exchanges and boundary conditions where they
/// are needed
///
/// Whenever a stencil writes to a variable and has and an other stencil has an  off-centered read
/// of the same variable afterwards, we need to do a halo exchange of the newly computed data
/// between these two stencils.
///
/// After this pass ran, the StencilDescAST will have boundary condition calls inserted at the
/// required spots such that whenever data needs to be transfered, Halo-Exchanges as well as
/// Boundary Conditions are triggered.
///
/// If a variable requires a boundary condition but no boundary conditon is specified, we issue an
/// assert during compilation
///
/// This pass should run after all the spltting is done: It depends on `PassStencilSplitter`
///
/// This pass is disabled until boundary conditions are redesigned.
///
/// @ingroup optimizer
///
/// This pass is not necessary to create legal code and is hence not in the debug-group
class PassSetBoundaryCondition : public Pass {
public:
  PassSetBoundaryCondition(OptimizerContext& context);

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) override;

private:
  std::vector<int> boundaryConditionInserted_;

  /// A map of all the stencils that a given field already applied its boundary conditons to
  std::unordered_map<std::string, std::vector<int>> StencilBCsApplied_;
};

} // namespace dawn
