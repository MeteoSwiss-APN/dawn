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

#include "dawn/Optimizer/PassStencilSplitter.h"
#include "dawn/Optimizer/DependencyGraphStage.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassSetStageGraph.h"
#include "dawn/Optimizer/PassTemporaryType.h"
#include "dawn/Optimizer/Replacing.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include <iostream>

namespace dawn {

/// @brief Check if we can merge `stage` into the stencil (given by `fields`) such that the number
/// of fields does not exceed `maxNumFields`
///
/// If the number of fields is already higher than `maxNumFields` we check if by merging stage into
/// the stencil we do not increase the number of fields further.
static int mergePossible(const std::set<int>& fields, const Stage* stage, int maxNumFields) {
  int numFields = fields.size();

  for(const Field& field : stage->getFields())
    if(!fields.count(field.getAccessID()))
      numFields++;

  // Inserting the stage would further increase the number of fields
  if(fields.size() > maxNumFields)
    return numFields == fields.size();

  return numFields <= maxNumFields;
}

PassStencilSplitter::PassStencilSplitter(int maxNumberOfFilelds)
    : Pass("PassStencilSplitter"), MaxFieldPerStencil(maxNumberOfFilelds) {
  dependencies_.push_back("PassSetStageGraph");
}

bool PassStencilSplitter::run(const std::shared_ptr<StencilInstantiation>& stencilInstantiation) {
  OptimizerContext* context = stencilInstantiation->getOptimizerContext();

  if(!context->getOptions().SplitStencils)
    return true;

  // If we split a stencil, we need to recompute the stage graphs
  bool rerunPassSetStageGraph = false;

  for(auto stencilIt = stencilInstantiation->getStencils().begin();
      stencilIt != stencilInstantiation->getStencils().end(); ++stencilIt) {
    Stencil& stencil = **stencilIt;

    // New stencils which serve as a replacement for `stencil`
    std::vector<std::shared_ptr<Stencil>> newStencils;

    // If a stencil exceeds the threshold, we need to split it
    if(stencil.getFields().size() > MaxFieldPerStencil) {
      rerunPassSetStageGraph = true;

      newStencils.emplace_back(std::make_shared<Stencil>(
          *stencilInstantiation, stencil.getSIRStencil(), stencilInstantiation->nextUID()));
      std::shared_ptr<Stencil> newStencil = newStencils.back();

      std::set<int> fieldsInNewStencil;

      // Iterate the multi-stage of the old `stencil` and insert its stages into `newStencil`
      for(const auto& multiStagePtr : stencil.getMultiStages()) {
        MultiStage& multiStage = *multiStagePtr;

        // Create an empty multi-stage in the current stencil with the same parameter as
        // `multiStage`
        newStencil->getMultiStages().push_back(
            std::make_shared<MultiStage>(*stencilInstantiation, multiStage.getLoopOrder()));

        for(std::shared_ptr<Stage>& stagePtr : multiStage.getStages()) {
          if(newStencil->isEmpty() ||
             mergePossible(fieldsInNewStencil, stagePtr.get(), MaxFieldPerStencil)) {

            // We can safely insert the stage into the current multi-stage of the `newStencil`
            newStencil->getMultiStages().back()->getStages().push_back(stagePtr);

            // Update fields of the `newStencil`. Note that the indivudual stages do not need to
            // update their fields as they remain the same.
            for(const Field& field : stagePtr->getFields())
              fieldsInNewStencil.insert(field.getAccessID());

          } else {
            // Make a new stencil
            newStencils.emplace_back(std::make_shared<Stencil>(
                *stencilInstantiation, stencil.getSIRStencil(), stencilInstantiation->nextUID()));
            newStencil = newStencils.back();
            fieldsInNewStencil.clear();

            // Re-create the current multi-stage in the `newStencil` and insert the stage
            newStencil->getMultiStages().push_back(
                std::make_shared<MultiStage>(*stencilInstantiation, multiStage.getLoopOrder()));
            newStencil->getMultiStages().back()->getStages().push_back(stagePtr);
          }
        }
      }
    }

    // Replace the current stencil with the new stencils and update the stencil description AST
    if(!newStencils.empty()) {

      // Repair broken references to temporaries i.e promote them to real fields
      PassTemporaryType::fixTemporariesSpanningMultipleStencils(stencilInstantiation.get(),
                                                                newStencils);

      // Remove empty multi-stages within the stencils
      for(auto& s : newStencils) {
        for(auto msIt = s->getMultiStages().begin(); msIt != s->getMultiStages().end();)
          if((*msIt)->isEmpty())
            msIt = s->getMultiStages().erase(msIt);
          else
            ++msIt;
      }

      // Update the fields of the stencil (this is not strictly necessary but it might be a source
      // of error in the future when updateFields also changes data-structures in the
      // Multi-Stage/Stencil)
      for(auto& s : newStencils)
        s->updateFields();

      // Update the calls to this stencil in the stencil description AST
      std::vector<int> newStencilIDs;
      for(const auto& s : newStencils)
        newStencilIDs.push_back(s->getStencilID());

      replaceStencilCalls(stencilInstantiation, stencil.getStencilID(), newStencilIDs);

      // Erase the old stencil ...
      stencilIt = stencilInstantiation->getStencils().erase(stencilIt);

      // ... and insert the new ones
      stencilIt = stencilInstantiation->getStencils().insert(stencilIt, newStencils.begin(),
                                                             newStencils.end());
      std::advance(stencilIt, newStencils.size() - 1);
    }
  }

  // Recompute the stage graph of each stencil
  if(rerunPassSetStageGraph) {
    PassSetStageGraph pass;
    pass.run(stencilInstantiation);
  }

  return true;
}

} // namespace dawn
