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

#include "dawn/Optimizer/PassIntegrityCheck.h"
#include "dawn/Validator/IntegrityChecker.h"

namespace dawn {

PassIntegrityCheck::PassIntegrityCheck(OptimizerContext& context)
    : Pass(context, "PassIntegrityCheck") {}

bool PassIntegrityCheck::run(const std::shared_ptr<iir::StencilInstantiation>& instantiation) {
  IntegrityChecker checker(instantiation.get());
  checker.run();
  return true;
}

} // namespace dawn
