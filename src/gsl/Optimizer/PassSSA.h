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

#ifndef GSL_OPTIMIZER_PASSSSA_H
#define GSL_OPTIMIZER_PASSSSA_H

#include "gsl/Optimizer/Pass.h"

namespace gsl {

/// @brief Converts each DAG of a stencil into SSA form (Static Single Assignment)
///
/// @see https://en.wikipedia.org/wiki/Static_single_assignment_form
/// @ingroup optimizer
class PassSSA : public Pass {
public:
  PassSSA();

  /// @brief Pass implementation
  bool run(StencilInstantiation* stencilInstantiation) override;
};

} // namespace gsl

#endif
