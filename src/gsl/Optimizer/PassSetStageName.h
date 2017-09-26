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

#ifndef GSL_OPTIMIZER_PASSSETSTAGENAME_H
#define GSL_OPTIMIZER_PASSSETSTAGENAME_H

#include "gsl/Optimizer/Pass.h"

namespace gsl {

/// @brief This pass assigns a unique name to each stage and makes
/// `StencilInstantiation::getNameFromStageID` usable
///
/// @ingroup optimizer
class PassSetStageName : public Pass {
public:
  PassSetStageName();

  /// @brief Pass implementation
  bool run(StencilInstantiation* stencilInstantiation) override;
};

} // namespace gsl

#endif
