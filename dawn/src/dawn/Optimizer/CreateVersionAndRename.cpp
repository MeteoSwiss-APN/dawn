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
#include "dawn/Optimizer/CreateVersionAndRename.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/AccessComputation.h"
#include "dawn/Optimizer/Renaming.h"

namespace dawn {

int createVersionAndRename(iir::StencilInstantiation* instantiation, int AccessID,
                           iir::Stencil* stencil, int curStageIdx, int curStmtIdx,
                           std::shared_ptr<iir::Expr>& expr, RenameDirection dir) {
  int newAccessID = -1;
  if(instantiation->getMetaData().isAccessType(iir::FieldAccessType::FAT_Field, AccessID)) {
    if(instantiation->getMetaData().variableHasMultipleVersions(AccessID)) {
      // Field is already multi-versioned, append a new version
      const auto versions = instantiation->getMetaData().getVersionsOf(AccessID);

      // Set the second to last field to be a temporary (only the first and the last field will be
      // real storages, all other versions will be temporaries)
      int lastAccessID = versions->back();
      instantiation->getMetaData().moveRegisteredFieldTo(iir::FieldAccessType::FAT_StencilTemporary,
                                                         lastAccessID);

      // The field with version 0 contains the original name
      int originalID = instantiation->getMetaData()
                           .getFieldAccessMetadata()
                           .variableVersions_.getOriginalVersionOfAccessID(lastAccessID);
      const std::string& originalName =
          instantiation->getMetaData().getFieldNameFromAccessID(originalID);

      // Register the new field
      newAccessID = instantiation->getMetaData().insertAccessOfType(
          iir::FieldAccessType::FAT_InterStencilTemporary,
          originalName + "_" + std::to_string(versions->size()));

      // and register in field-versioning
      instantiation->getMetaData().addFieldVersionIDPair(originalID, newAccessID);

    } else {
      const std::string& originalName =
          instantiation->getMetaData().getFieldNameFromAccessID(AccessID);

      newAccessID = instantiation->getMetaData().insertAccessOfType(
          iir::FieldAccessType::FAT_InterStencilTemporary, originalName + "_0");

      // Register the new *and* old field as being multi-versioned and indicate code-gen it has to
      // allocate the second version
      instantiation->getMetaData().addFieldVersionIDPair(AccessID, newAccessID);
    }
  } else {
    // if not a field, it is a variable
    if(instantiation->getMetaData().variableHasMultipleVersions(AccessID)) {
      // Variable is already multi-versioned, append a new version
      auto versions = instantiation->getMetaData().getVersionsOf(AccessID);

      int lastAccessID = versions->back();
      // The field with version 0 contains the original name
      int originalID = instantiation->getMetaData()
                           .getFieldAccessMetadata()
                           .variableVersions_.getOriginalVersionOfAccessID(lastAccessID);
      const std::string& originalName =
          instantiation->getMetaData().getFieldNameFromAccessID(originalID);

      // Register the new variable
      newAccessID = instantiation->getMetaData().insertAccessOfType(
          iir::FieldAccessType::FAT_LocalVariable,
          originalName + "_" + std::to_string(versions->size()));

      instantiation->getMetaData().addFieldVersionIDPair(originalID, newAccessID);

    } else {
      const std::string& originalName =
          instantiation->getMetaData().getFieldNameFromAccessID(AccessID);

      newAccessID = instantiation->getMetaData().insertAccessOfType(
          iir::FieldAccessType::FAT_LocalVariable, originalName + "_0");
      // Register the new *and* old variable as being multi-versioned
      instantiation->getMetaData().addFieldVersionIDPair(AccessID, newAccessID);
    }
  }

  // Rename the Expression
  renameAccessIDInExpr(instantiation, AccessID, newAccessID, expr);

  // Recompute the accesses of the current statement (only works with single Do-Methods - for now)
  computeAccesses(
      instantiation,
      stencil->getStage(curStageIdx)->getSingleDoMethod().getAST().getStatements()[curStmtIdx]);

  // Rename the statement and accesses
  for(int stageIdx = curStageIdx;
      dir == RenameDirection::Above ? (stageIdx >= 0) : (stageIdx < stencil->getNumStages());
      dir == RenameDirection::Above ? stageIdx-- : stageIdx++) {
    iir::Stage& stage = *stencil->getStage(stageIdx);
    iir::DoMethod& doMethod = stage.getSingleDoMethod();

    if(stageIdx == curStageIdx) {
      for(int i = dir == RenameDirection::Above ? (curStmtIdx - 1) : (curStmtIdx + 1);
          dir == RenameDirection::Above ? (i >= 0) : (i < doMethod.getAST().getStatements().size());
          dir == RenameDirection::Above ? (--i) : (++i)) {
        renameAccessIDInStmts(&instantiation->getMetaData(), AccessID, newAccessID,
                              doMethod.getAST().getStatements()[i]);
        renameAccessIDInAccesses(&instantiation->getMetaData(), AccessID, newAccessID,
                                 doMethod.getAST().getStatements()[i]);
      }

    } else {
      renameAccessIDInStmts(&instantiation->getMetaData(), AccessID, newAccessID,
                            doMethod.getAST().getStatements());
      renameAccessIDInAccesses(&instantiation->getMetaData(), AccessID, newAccessID,
                               doMethod.getAST().getStatements());
    }

    // Update the fields of the doMethod and stage levels
    doMethod.update(iir::NodeUpdateType::level);
    stage.update(iir::NodeUpdateType::level);
  }
  return newAccessID;
}
} // namespace dawn