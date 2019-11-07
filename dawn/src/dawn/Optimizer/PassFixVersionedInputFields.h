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

#ifndef DAWN_OPTIMIZER_PASSFIXFIELDVERSIONEDFIELDS_H
#define DAWN_OPTIMIZER_PASSFIXFIELDVERSIONEDFIELDS_H

#include "dawn/Optimizer/Pass.h"

namespace dawn {

/// @brief This pass ensures that if input-fields are versioned and the version is used first with a
/// read-access that is is filled with the correct input data beforehand
///
/// @ingroup optimizer
class PassFixVersionedInputFields : public Pass {
public:
  PassFixVersionedInputFields(OptimizerContext& context);

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) override;
};

} // namespace dawn

#endif
