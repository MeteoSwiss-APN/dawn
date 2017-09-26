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

#ifndef GSL_OPTIMIZER_PASSTEMPORARYMERGER_H
#define GSL_OPTIMIZER_PASSTEMPORARYMERGER_H

#include "gsl/Optimizer/Pass.h"

namespace gsl {

/// @brief Pass to merge temporaries
/// @ingroup optimizer
class PassTemporaryMerger : public Pass {
public:
  PassTemporaryMerger();

  /// @brief Pass implementation
  bool run(StencilInstantiation* stencilInstantiation) override;
};

} // namespace gsl

#endif
