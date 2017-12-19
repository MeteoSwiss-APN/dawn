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
/// Whenever a stencil writes to a variable and has an off-centered read of the same variable
/// afterwards, we need to do a halo exchange of the newly computed data. If the boundary condition
/// that we use for this variable has read-access to itself (most notably zero-gradient), we need to
/// apply this boundary condition once more.
///
/// After this pass ran, stages will be inserted into the ISIR that only have boundary condition
/// calls inside since they require synchronization.
///
/// This pass has no dependencies
///
/// @ingroup optimizer
class PassSetBoundaryCondition : public Pass {
public:
  PassSetBoundaryCondition();

  /// @brief Pass implementation
  bool run(StencilInstantiation* stencilInstantiation) override;
};

} // namespace dawn

#endif
