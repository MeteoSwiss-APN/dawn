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
#include "dawn/Validator/WeightChecker.h"

namespace dawn {

bool PassValidation::run(const std::shared_ptr<iir::StencilInstantiation>& instantiation,
                         const Options& options) {
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
    auto [dimsConsistent, dimsConsistencyErrorLocation] =
        UnstructuredDimensionChecker::checkDimensionsConsistency(*iir, metadata);
    DAWN_ASSERT_MSG(dimsConsistent,
                    ("Dimensions consistency check failed at line " +
                     std::to_string(dimsConsistencyErrorLocation.Line) + " " + description)
                        .c_str());
    auto [stageConsistent, stageConsistencyErrorLocation] =
        UnstructuredDimensionChecker::checkStageLocTypeConsistency(*iir, metadata);
    DAWN_ASSERT_MSG(stageConsistent,
                    ("Stage location type consistency check failed at line " +
                     std::to_string(stageConsistencyErrorLocation.Line) + " " + description)
                        .c_str());
    auto [weightsValid, weightValidErrorLocation] = WeightChecker::CheckWeights(*iir, metadata);
    DAWN_ASSERT_MSG(weightsValid,
                    ("Found invalid weights at line " +
                     std::to_string(weightValidErrorLocation.Line) + " " + description)
                        .c_str());
  }

  DAWN_ASSERT_MSG(GridTypeChecker::checkGridTypeConsistency(*iir),
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
    auto [checkResultDimensions, errorLocationDimensions] =
        UnstructuredDimensionChecker::checkDimensionsConsistency(*sir);
    DAWN_ASSERT_MSG(checkResultDimensions, ("Dimensions in SIR are not consistent at line " +
                                            std::to_string(errorLocationDimensions.Line))
                                               .c_str());
    auto [checkResultWeights, errorLocationWeights] =
        UnstructuredDimensionChecker::checkDimensionsConsistency(*sir);
    DAWN_ASSERT_MSG(
        checkResultWeights,
        ("Found invalid weights at line " + std::to_string(errorLocationWeights.Line)).c_str());
  }

  DAWN_ASSERT_MSG(GridTypeChecker::checkGridTypeConsistency(*sir),
                  "Grid types in SIR are not consistent");

  return true;
}

} // namespace dawn
