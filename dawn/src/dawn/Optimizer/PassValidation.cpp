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
  const auto& iir = instantiation->getIIR();
  const auto& metadata = instantiation->getMetaData();

  if(!iir->checkTreeConsistency())
    throw SemanticError("Tree consistency check failed " + description, metadata.getFileName());

  if(iir->getGridType() == ast::GridType::Unstructured) {
    UnstructuredDimensionChecker dimensionsChecker;
    DAWN_ASSERT_MSG(dimensionsChecker.checkDimensionsConsistency(*iir, metadata),
                    ("Dimensions consistency check failed " + description).c_str());
  }

  GridTypeChecker gridChecker;
  DAWN_ASSERT_MSG(gridChecker.checkGridTypeConsistency(*iir),
                  ("Grid type consistency check failed " + description).c_str());

  IntegrityChecker checker(instantiation.get());
  try {
    checker.run();
  } catch(CompileError& error) {
    DAWN_ASSERT_MSG(false, error.getMessage().c_str());
  }
  return true;
}

bool PassValidation::run(const std::shared_ptr<dawn::SIR>& sir) {
  if(sir->GridType == ast::GridType::Unstructured) {
    UnstructuredDimensionChecker dimensionsChecker;
    DAWN_ASSERT_MSG(dimensionsChecker.checkDimensionsConsistency(*sir),
                    "Dimensions in SIR are not consistent");
  }

  GridTypeChecker gridChecker;
  DAWN_ASSERT_MSG(gridChecker.checkGridTypeConsistency(*sir),
                  "Grid types in SIR are not consistent");

  return true;
}

} // namespace dawn
