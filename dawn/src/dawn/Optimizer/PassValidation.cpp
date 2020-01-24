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

#include "dawn/Optimizer/PassValidation.h"

namespace dawn {

PassValidation::PassValidation(OptimizerContext& context) : Pass(context, "PassValidation") {}

bool PassValidation::run(const std::shared_ptr<iir::StencilInstantiation>& instantiation) {
  IntegrityChecker integrityChecker(instantiation.get());
  integrityChecker.run();

  GridTypeChecker gridTypeChecker;
  bool consistent = gridTypeChecker.checkGridTypeConsistency(*instantiation->getIIR());
  if(!consistent) {
    throw SemanticError("Grid types in IIR are not consistent");
  }

  LocationTypeChecker locationChecker;
  consistent = locationChecker.checkLocationTypeConsistency(*instantiation->getIIR(),
                                                            instantiation->getMetaData());
  if(!consistent) {
    throw SemanticError("Location types in IIR are not consistent");
  }

  return true;
}

} // namespace dawn
