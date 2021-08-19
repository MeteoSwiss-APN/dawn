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
#include "dawn/Validator/IndirectionChecker.h"
#include "dawn/Validator/IntegrityChecker.h"
#include "dawn/Validator/MultiStageChecker.h"
#include "dawn/Validator/UnstructuredDimensionChecker.h"
#include "dawn/Validator/WeightChecker.h"

namespace dawn {

bool PassValidation::run(const std::shared_ptr<iir::StencilInstantiation>& instantiation,
                         const Options& options) {
  return run(instantiation, options, "");
}

// TODO: explain what description is
bool PassValidation::run(const std::shared_ptr<iir::StencilInstantiation>& instantiation,
                         const Options& options, const std::string& description) {
  const auto& iir = instantiation->getIIR();
  const auto& metadata = instantiation->getMetaData();

  if(!iir->checkTreeConsistency())
    throw SemanticError("Tree consistency check failed " + description, metadata.getFileName());

  if(iir->getGridType() == ast::GridType::Unstructured) {
    auto [dimsConsistent, dimsConsistencyErrorLocation] =
        UnstructuredDimensionChecker::checkDimensionsConsistency(*iir, metadata);
    if(!dimsConsistent)
      throw SemanticError("Dimensions consistency check failed at line " +
                          std::to_string(dimsConsistencyErrorLocation.Line) + " " + description);
    auto [stageConsistent, stageConsistencyErrorLocation] =
        UnstructuredDimensionChecker::checkStageLocTypeConsistency(*iir, metadata);
    if(!stageConsistent)
      throw SemanticError("Stage location type consistency check failed at line " +
                          std::to_string(stageConsistencyErrorLocation.Line) + " " + description);
    auto [weightsValid, weightValidErrorLocation] = WeightChecker::CheckWeights(*iir, metadata);
    if(!weightsValid)
      throw SemanticError("Found invalid weights at line " +
                          std::to_string(weightValidErrorLocation.Line) + " " + description);
  }

  auto [indirectionsValid, indirectionsValidErrorLocation] =
      IndirectionChecker::checkIndirections(*iir);
  if(!indirectionsValid)
    throw SemanticError("Found invalid indirection at line " +
                        std::to_string(indirectionsValidErrorLocation.Line) + " " + description);

  if(!GridTypeChecker::checkGridTypeConsistency(*iir))
    throw SemanticError("Grid type consistency check failed " + description);

  IntegrityChecker integrityChecker(instantiation.get());
  integrityChecker.run();

  if(iir->getGridType() != ast::GridType::Unstructured) {
    MultiStageChecker multiStageChecker;
    multiStageChecker.run(instantiation.get(), options.MaxHaloPoints);
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
    if(!checkResultDimensions)
      throw SemanticError("Dimensions in SIR are not consistent at line " +
                          std::to_string(errorLocationDimensions.Line));
    
    auto [checkResultWeights, errorLocationWeights] =
        UnstructuredDimensionChecker::checkDimensionsConsistency(*sir);
    if(!checkResultWeights)
      throw SemanticError("Found invalid weights at line " +
                          std::to_string(errorLocationWeights.Line));
  }

  auto [indirectionsValid, indirectionsValidErrorLocation] =
      IndirectionChecker::checkIndirections(*sir);
  if(!indirectionsValid)
    throw SemanticError("Found invalid indirection at line " +
                        std::to_string(indirectionsValidErrorLocation.Line));

  if(!GridTypeChecker::checkGridTypeConsistency(*sir))
    throw SemanticError("Grid types in SIR are not consistent");

  return true;
}

} // namespace dawn
