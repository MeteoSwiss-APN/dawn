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

#include "dawn/Optimizer/PassStageReordering.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Support/FileSystem.h"
#include "dawn/Support/Unreachable.h"

#include "dawn/Optimizer/ReorderStrategyGreedy.h"
#include "dawn/Optimizer/ReorderStrategyPartitioning.h"

namespace dawn {

bool PassStageReordering::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const Options& options) {
  const std::string filenameWE =
      fs::path(stencilInstantiation->getMetaData().getFileName()).filename().stem();

  if(options.WriteStencilInstantiation)
    stencilInstantiation->jsonDump(filenameWE + "_before_stage_reordering.json");

  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
    if(strategy_ == ReorderStrategy::Kind::None)
      continue;

    std::unique_ptr<ReorderStrategy> strategy;
    switch(strategy_) {
    case ReorderStrategy::Kind::Greedy:
      strategy = std::make_unique<ReoderStrategyGreedy>();
      break;
    case ReorderStrategy::Kind::Partitioning:
      strategy = std::make_unique<ReoderStrategyPartitioning>();
      break;
    default:
      dawn_unreachable("invalid reorder strategy");
    }

    // TODO should we have Iterators so to prevent unique_ptr swaps
    auto newStencil = strategy->reorder(stencilInstantiation.get(), stencilPtr, options);

    stencilInstantiation->getIIR()->replace(stencilPtr, newStencil, stencilInstantiation->getIIR());
    // TODO why is stencilPtr still used? (it was replaced in the previous statement)
    stencilPtr->update(iir::NodeUpdateType::levelAndTreeAbove);

    if(!stencilPtr)
      return false;
  }

  if(options.WriteStencilInstantiation)
    stencilInstantiation->jsonDump(filenameWE + "_after_stage_reordering.json");

  return true;
}

} // namespace dawn
