//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#ifndef GSL_OPTIMIZER_PASSPRINTSTENCILGRAPH_H
#define GSL_OPTIMIZER_PASSPRINTSTENCILGRAPH_H

#include "gsl/Optimizer/Pass.h"

namespace gsl {

/// @brief This Pass prints the dependency graph of each stencil to a dot file
///
/// This Pass depends on `PassStageSplitter` (which sets the dependency graphs).
///
/// @ingroup optimizer
class PassPrintStencilGraph : public Pass {
public:
  PassPrintStencilGraph();

  /// @brief Pass implementation
  bool run(StencilInstantiation* stencilInstantiation) override;
};

} // namespace gsl

#endif
