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

#ifndef DAWN_OPTIMIZER_PASSSETBOUNDARYCONDITIONS_H
#define DAWN_OPTIMIZER_PASSSETBOUNDARYCONDITIONS_H

#include "dawn/Optimizer/Pass.h"

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
/// This pass has no dependencies
///
/// @ingroup optimizer
class PassSetBoundaryCondition : public Pass {
public:
  PassSetBoundaryCondition();

  /// @brief Pass implementation
  bool run(StencilInstantiation* stencilInstantiation) override;

private:
  std::vector<int> boundaryConditionInserted_;
};

} // namespace dawn

#endif
