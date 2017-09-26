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

#ifndef GSL_OPTIMIZER_PASSTEMPORARYFIRSTACCESS_H
#define GSL_OPTIMIZER_PASSTEMPORARYFIRSTACCESS_H

#include "gsl/Optimizer/Pass.h"

namespace gsl {

/// @brief This pass checks the first access to a temporary to avoid unitialized memory accesses
///
/// @ingroup optimizer
class PassTemporaryFirstAccess : public Pass {
public:
  PassTemporaryFirstAccess();

  /// @brief Pass implementation
  bool run(StencilInstantiation* stencilInstantiation) override;
};

} // namespace gsl

#endif
