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

#ifndef DAWN_OPTIMIZER_PASSTEMPORARYTOSTENCILFUNCTION_H
#define DAWN_OPTIMIZER_PASSTEMPORARYTOSTENCILFUNCTION_H

#include "dawn/Optimizer/Pass.h"

namespace dawn {

class Stencil;
class DoMethod;

/// @brief PassTemporaryToStencilFunction
/// @ingroup optimizer
class PassTemporaryToStencilFunction : public Pass {

public:
  PassTemporaryToStencilFunction();

  /// @brief Pass implementation
  bool run(std::shared_ptr<StencilInstantiation> stencilInstantiation) override;
};

} // namespace dawn

#endif
