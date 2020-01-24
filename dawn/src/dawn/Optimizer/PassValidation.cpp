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
  bool consistent = true;
  IntegrityChecker checker(instantiation.get());
  checker.run();

  const auto& metaData = instantiation->getMetaData();
  const auto& iir = instantiation->getIIR();

  if(iir->getGridType() == ast::GridType::Triangular) {
    LocationTypeChecker locationChecker;
    consistent = locationChecker.checkLocationTypeConsistency(*iir, metaData);
    if(!consistent)
      throw SemanticError("Location types in IIR are not consistent", metaData.getFileName());
  }

  GridTypeChecker gridChecker;
  consistent = gridChecker.checkGridTypeConsistency(*iir);
  if(!consistent)
    throw SemanticError("Grid types in IIR are not consistent");

  return consistent;
}

} // namespace dawn
