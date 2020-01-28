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
  const auto& metadata = instantiation->getMetaData();

  if(!iir->checkTreeConsistency())
    throw SemanticError("Tree consistency check failed " + description, metadata.getFileName());

  if(iir->getGridType() == ast::GridType::Unstructured) {
    UnstructuredDimensionChecker dimensionsChecker;
    if(!dimensionsChecker.checkDimensionsConsistency(*iir, metadata))
      throw SemanticError("Dimensions consistency check failed " + description,
                          metadata.getFileName());
  }

  GridTypeChecker gridChecker;
  if(!gridChecker.checkGridTypeConsistency(*iir))
    throw SemanticError("Grid type consistency check failed " + description,
                        metadata.getFileName());

  return true;
}

bool PassValidation::run(const std::shared_ptr<dawn::SIR>& sir) {
  if(sir->GridType == ast::GridType::Unstructured) {
    UnstructuredDimensionChecker dimensionsChecker;
    if(!dimensionsChecker.checkDimensionsConsistency(*sir))
      throw SemanticError("Dimension types in SIR are not consistent", sir->Filename);
  }

  GridTypeChecker gridChecker;
  if(!gridChecker.checkGridTypeConsistency(*sir))
    throw SemanticError("Grid types in SIR are not consistent", sir->Filename);

  return true;
}

} // namespace dawn
