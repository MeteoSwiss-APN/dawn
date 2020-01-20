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

#ifndef DAWN_OPTIMIZER_PASSSETDEPENDENCYGRAPH_H
#define DAWN_OPTIMIZER_PASSSETDEPENDENCYGRAPH_H

#include "dawn/Optimizer/Pass.h"

namespace dawn {

/// @brief This Pass sets the dependency graph of a do-method
class PassSetDependencyGraph : public Pass {
public:
  PassSetDependencyGraph(OptimizerContext& context);

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) override;
};

} // namespace dawn

#endif // DAWN_OPTIMIZER_PASSSETDEPENDENCYGRAPH_H
