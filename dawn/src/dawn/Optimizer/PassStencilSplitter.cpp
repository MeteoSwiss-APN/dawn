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
#include "dawn/IIR/DependencyGraphStage.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassSetStageGraph.h"
#include "dawn/Optimizer/PassTemporaryType.h"
#include "dawn/Optimizer/Replacing.h"

namespace dawn {

/// @brief Check if we can merge `stage` into the stencil (given by `fields`) such that the number
/// of fields does not exceed `maxNumFields`
///
/// If the number of fields is already higher than `maxNumFields` we check if by merging stage into
/// the stencil we do not increase the number of fields further.
static int mergePossible(const std::set<int>& fields, const iir::Stage* stage, int maxNumFields) {
  int numFields = fields.size();

  for(const auto& fieldPair : stage->getFields())
    if(!fields.count(fieldPair.second.getAccessID()))
      numFields++;

  // Inserting the stage would further increase the number of fields
  if(fields.size() > maxNumFields)
    return numFields == fields.size();

  return numFields <= maxNumFields;
}

PassStencilSplitter::PassStencilSplitter(OptimizerContext& context, int maxNumberOfFilelds)
    : Pass(context, "PassStencilSplitter"), MaxFieldPerStencil(maxNumberOfFilelds) {
  dependencies_.push_back("PassSetStageGraph");
}

bool PassStencilSplitter::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  if(!context_.getOptions().SplitStencils)
    return true;

  // If we split a stencil, we need to recompute the stage graphs
  bool rerunPassSetStageGraph = false;

  for(auto stencilIt = stencilInstantiation->getIIR()->childrenBegin();
      stencilIt != stencilInstantiation->getIIR()->childrenEnd(); ++stencilIt) {

    iir::Stencil& stencil = **stencilIt;

    // New stencils which serve as a replacement for `stencil`
    std::vector<std::unique_ptr<iir::Stencil>> newStencils;

    // If a stencil exceeds the threshold, we need to split it
    if(stencil.getFields().size() > MaxFieldPerStencil) {
      rerunPassSetStageGraph = true;

      newStencils.emplace_back(std::make_unique<iir::Stencil>(stencilInstantiation->getMetaData(),
                                                              stencil.getStencilAttributes(),
                                                              stencilInstantiation->nextUID()));
      const std::unique_ptr<iir::Stencil>& newStencil = newStencils.back();

      std::set<int> fieldsInNewStencil;

      // Iterate the multi-stage of the old `stencil` and insert its stages into `newStencil`
      for(const auto& multiStagePtr : stencil.getChildren()) {
        iir::MultiStage& multiStage = *multiStagePtr;

        // Create an empty multi-stage in the current stencil with the same parameter as
        // `multiStage`
        newStencil->insertChild(std::make_unique<iir::MultiStage>(
            stencilInstantiation->getMetaData(), multiStage.getLoopOrder()));

        for(const auto& stagePtr : multiStage.getChildren()) {
          if(newStencil->isEmpty() ||
             mergePossible(fieldsInNewStencil, stagePtr.get(), MaxFieldPerStencil)) {

            // We can safely insert the stage into the current multi-stage of the `newStencil`
            newStencil->getChildren().back()->insertChild(stagePtr->clone());

            // Update fields of the `newStencil`. Note that the indivudual stages do not need to
            // update their fields as they remain the same.
            for(const auto& fieldPair : stagePtr->getFields())
              fieldsInNewStencil.insert(fieldPair.second.getAccessID());

          } else {
            // Make a new stencil
            newStencils.emplace_back(std::make_unique<iir::Stencil>(
                stencilInstantiation->getMetaData(), stencil.getStencilAttributes(),
                stencilInstantiation->nextUID()));
            const std::unique_ptr<iir::Stencil>& newStencil2 = newStencils.back();

            fieldsInNewStencil.clear();

            // Re-create the current multi-stage in the `newStencil` and insert the stage
            newStencil2->insertChild(std::make_unique<iir::MultiStage>(
                stencilInstantiation->getMetaData(), multiStage.getLoopOrder()));
            newStencil2->getChildren().back()->insertChild(stagePtr->clone());
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
        for(auto msIt = s->childrenBegin(); msIt != s->childrenEnd();)
          if((*msIt)->childrenEmpty())
            msIt = s->childrenErase(msIt);
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
      stencilIt = stencilInstantiation->getIIR()->childrenErase(stencilIt);

      // ... and insert the new ones
      stencilIt = stencilInstantiation->getIIR()->insertChildren(
          stencilIt, std::make_move_iterator(newStencils.begin()),
          std::make_move_iterator(newStencils.end()), stencilInstantiation->getIIR());
      std::advance(stencilIt, newStencils.size() - 1);
    }
  }

  // Recompute the stage graph of each stencil
  if(rerunPassSetStageGraph) {
    PassSetStageGraph pass(context_);
    pass.run(stencilInstantiation);
  }

  return true;
}

} // namespace dawn
