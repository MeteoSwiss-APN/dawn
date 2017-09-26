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

#ifndef GSL_OPTIMIZER_PASSSETSTAGEGRAPH_H
#define GSL_OPTIMIZER_PASSSETSTAGEGRAPH_H

#include "gsl/Optimizer/Pass.h"

namespace gsl {

/// @brief This Pass computes and assign the stage graph of each stencil
///
/// This Pass depends on `PassSetStageName`.
///
/// @ingroup optimizer
class PassSetStageGraph : public Pass {
public:
  PassSetStageGraph();

  /// @brief Pass implementation
  bool run(StencilInstantiation* stencilInstantiation) override;
};

} // namespace gsl

#endif
