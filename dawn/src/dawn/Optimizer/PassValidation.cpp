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
  return run(instantiation, "");
}

bool PassValidation::run(const std::shared_ptr<iir::StencilInstantiation>& instantiation,
                         const std::string& description) {
  IntegrityChecker checker(instantiation.get());
  checker.run();

  const auto& metaData = instantiation->getMetaData();
  const auto& iir = instantiation->getIIR();

  if(!iir->checkTreeConsistency()) {
    DAWN_LOG(WARNING) << "Tree consistency check failed " << description;
    return false;
  }

  if(iir->getGridType() == ast::GridType::Triangular) {
    LocationTypeChecker locationChecker;
    if(!locationChecker.checkLocationTypeConsistency(*iir, metaData)) {
      DAWN_LOG(WARNING) << "Location type consistency check failed " << description;
      return false;
    }
  }

  GridTypeChecker gridChecker;
  if(!gridChecker.checkGridTypeConsistency(*iir)) {
    DAWN_LOG(WARNING) << "Type consistency check failed " << description;
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
