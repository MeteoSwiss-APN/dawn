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

#ifndef GSL_OPTIMIZER_PASSSETMULTISTAGECACHES_H
#define GSL_OPTIMIZER_PASSSETMULTISTAGECACHES_H

#include "gsl/Optimizer/Pass.h"

namespace gsl {

/// @brief Determine which fields can be cached during the executation of the multi-stage
///
/// @ingroup optimizer
class PassSetCaches : public Pass {
public:
  PassSetCaches();

  /// @brief Pass implementation
  bool run(StencilInstantiation* stencilInstantiation) override;
};

} // namespace gsl

#endif
