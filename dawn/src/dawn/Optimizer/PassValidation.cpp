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

  const auto& iir = instantiation->getIIR();

  DAWN_ASSERT_MSG(iir->checkTreeConsistency(),
                  (std::string("Tree consistency check failed ") + description).c_str());

  if(iir->getGridType() == ast::GridType::Unstructured) {
    UnstructuredDimensionChecker dimensionsChecker;
    DAWN_ASSERT_MSG(
        dimensionsChecker.checkDimensionsConsistency(*iir, instantiation->getMetaData()),
        (std::string("Dimensions consistency check failed ") + description).c_str());
  }

  GridTypeChecker gridChecker;
  DAWN_ASSERT_MSG(gridChecker.checkGridTypeConsistency(*iir),
                  (std::string("Type consistency check failed ") + description).c_str());

  return true;
}

bool PassValidation::run(const std::shared_ptr<dawn::SIR>& sir) {
  if(sir->GridType == ast::GridType::Unstructured) {
    UnstructuredDimensionChecker dimensionsChecker;
    if(!dimensionsChecker.checkDimensionsConsistency(*sir)) {
      DAWN_LOG(WARNING) << "Dimension types in SIR are not consistent";
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
