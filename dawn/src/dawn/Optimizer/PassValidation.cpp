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

bool PassValidation::run(const std::shared_ptr<dawn::SIR>& sir) {
  // SIR we received should be type consistent
  if(sir->GridType == ast::GridType::Triangular) {
    LocationTypeChecker locationChecker;
    if(!locationChecker.checkLocationTypeConsistency(*sir))
      throw SemanticError("Location types in SIR are not consistent");
  }

  GridTypeChecker gridChecker;
  if(!gridChecker.checkGridTypeConsistency(*sir))
    throw SemanticError("Grid types in SIR are not consistent");

  return true;
}

} // namespace dawn
