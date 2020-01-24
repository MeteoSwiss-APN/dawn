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
  IntegrityChecker checker(instantiation.get());
  try {
    checker.run();
  } catch(CompileError& error) {
    DAWN_LOG(WARNING) << error.getMessage();
    return false;
  }

  const auto& metaData = instantiation->getMetaData();
  const auto& iir = instantiation->getIIR();

  if(iir->getGridType() == ast::GridType::Triangular) {
    LocationTypeChecker locationChecker;
    if(!locationChecker.checkLocationTypeConsistency(*iir, metaData)) {
      DAWN_LOG(WARNING) << "Location types in IIR are not consistent in '" << metaData.getFileName() << "'";
      return false;
    }
  }

  GridTypeChecker gridChecker;
  if(!gridChecker.checkGridTypeConsistency(*iir)) {
    DAWN_LOG(WARNING) << "Grid types in IIR are not consistent in '" << metaData.getFileName() << "'";
    return false;
  }

  return true;
}

bool PassValidation::run(const std::shared_ptr<dawn::SIR>& sir) {
  if(sir->GridType == ast::GridType::Triangular) {
    LocationTypeChecker locationChecker;
    if(!locationChecker.checkLocationTypeConsistency(*sir)) {
      DAWN_LOG(WARNING) << "Location types in SIR are not consistent";
      return false;
    }
  }

  GridTypeChecker gridChecker;
  if(!gridChecker.checkGridTypeConsistency(*sir)) {
    DAWN_LOG(WARNING) << "Grid types in SIR are not consistent";
    return false;
  }

  return true;
}

} // namespace dawn
