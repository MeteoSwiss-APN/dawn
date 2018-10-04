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

#ifndef DAWN_OPTIMIZER_PASSTEMPORARYFIRSTACCESS_H
#define DAWN_OPTIMIZER_PASSTEMPORARYFIRSTACCESS_H

#include "dawn/Optimizer/Pass.h"

namespace dawn {

/// @brief This pass checks the first access to a temporary to avoid unitialized memory accesses
///
/// @ingroup optimizer
class PassTemporaryFirstAccess : public Pass {
public:
  PassTemporaryFirstAccess();

  /// @brief Pass implementation
  bool run(const std::unique_ptr<iir::IIR>& iir) override;
};

} // namespace dawn

#endif
