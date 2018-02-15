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

#ifndef DAWN_OPTIMIZER_PASSSETSTAGENAME_H
#define DAWN_OPTIMIZER_PASSSETSTAGENAME_H

#include "dawn/Optimizer/Pass.h"

namespace dawn {

/// @brief This pass assigns a unique name to each stage and makes
/// `StencilInstantiation::getNameFromStageID` usable
///
/// @ingroup optimizer
class PassSetStageName : public Pass {
public:
  PassSetStageName();

  /// @brief Pass implementation
  bool run(const std::shared_ptr<StencilInstantiation>& stencilInstantiation) override;
};

} // namespace dawn

#endif
