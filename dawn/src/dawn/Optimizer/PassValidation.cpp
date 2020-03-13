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
#include "dawn/Support/Exception.h"
#include "dawn/Validator/GridTypeChecker.h"
#include "dawn/Validator/IntegrityChecker.h"
#include "dawn/Validator/UnstructuredDimensionChecker.h"

namespace dawn {

PassValidation::PassValidation(OptimizerContext& context) : Pass(context, "PassValidation") {}

bool PassValidation::run(const std::shared_ptr<iir::StencilInstantiation>& instantiation) {
  return run(instantiation, "");
}

// TODO: explain what description is
bool PassValidation::run(const std::shared_ptr<iir::StencilInstantiation>& instantiation,
                         const std::string& description) {
  const auto& iir = instantiation->getIIR();
  const auto& metadata = instantiation->getMetaData();

  try {
    if(!iir->checkTreeConsistency())
      throw SemanticError("Tree consistency check failed " + description, metadata.getFileName());
  } catch(CompileError& error) {
    DAWN_ASSERT_MSG(false, error.getMessage().c_str());
  }

  if(iir->getGridType() == ast::GridType::Unstructured) {
    UnstructuredDimensionChecker dimensionsChecker;
    auto [dimsConsistent, dimsConsistencyErrorLocation] =
        dimensionsChecker.checkDimensionsConsistency(*iir, metadata);
    DAWN_ASSERT_MSG(dimsConsistent,
                    ("Dimensions consistency check failed at line " +
                     std::to_string(dimsConsistencyErrorLocation.Line) + " " + description)
                        .c_str());
    auto [stageConsistent, stageConsistencyErrorLocation] =
        dimensionsChecker.checkStageLocTypeConsistency(*iir, metadata);
    DAWN_ASSERT_MSG(stageConsistent,
                    ("Stage location type consistency check failed at line " +
                     std::to_string(stageConsistencyErrorLocation.Line) + " " + description)
                        .c_str());
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

#ifndef NDEBUG
  for(const auto& stencil : instantiation->getIIR()->getChildren()) {
    DAWN_ASSERT(stencil->compareDerivedInfo());
  }
#endif
  return true;
}

bool PassValidation::run(const std::shared_ptr<dawn::SIR>& sir) {
  if(sir->GridType == ast::GridType::Unstructured) {
    UnstructuredDimensionChecker dimensionsChecker;
    auto [checkResult, errorLocation] = dimensionsChecker.checkDimensionsConsistency(*sir);
    DAWN_ASSERT_MSG(checkResult, ("Dimensions in SIR are not consistent at line " +
                                  std::to_string(errorLocation.Line))
                                     .c_str());
  }

  GridTypeChecker gridChecker;
  DAWN_ASSERT_MSG(gridChecker.checkGridTypeConsistency(*sir),
                  "Grid types in SIR are not consistent");

  return true;
}

} // namespace dawn
