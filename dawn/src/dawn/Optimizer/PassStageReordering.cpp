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
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/ReorderStrategyGreedy.h"
#include "dawn/Optimizer/ReorderStrategyPartitioning.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Support/FileSystem.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {

PassStageReordering::PassStageReordering(OptimizerContext& context, ReorderStrategy::Kind strategy)
    : Pass(context, "PassStageReordering"), strategy_(strategy) {
  dependencies_.push_back("PassSetStageGraph");
}

bool PassStageReordering::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  const std::string filenameWE =
      fs::path(stencilInstantiation->getMetaData().getFileName()).filename().stem();
  if(context_.getOptions().ReportPassStageReordering)
    stencilInstantiation->jsonDump(filenameWE + "_before.json");

  for(const auto& stencil : stencilInstantiation->getStencils()) {
    if(strategy_ == ReorderStrategy::Kind::None)
      continue;

    std::unique_ptr<ReorderStrategy> strategy;
    switch(strategy_) {
    case ReorderStrategy::Kind::Greedy:
      strategy = std::make_unique<ReorderStrategyGreedy>();
      break;
    case ReorderStrategy::Kind::Partitioning:
      strategy = std::make_unique<ReorderStrategyPartitioning>();
      break;
    default:
      dawn_unreachable("invalid reorder strategy");
    }

    // TODO should we have Iterators so to prevent unique_ptr swaps
    auto newStencil = strategy->reorder(stencilInstantiation.get(), stencil, context_);
    stencilInstantiation->getIIR()->replace(stencil, newStencil, stencilInstantiation->getIIR());
    newStencil->update(iir::NodeUpdateType::levelAndTreeAbove);

    if(!newStencil)
      return false;
  }

  if(context_.getOptions().ReportPassStageReordering)
    stencilInstantiation->jsonDump(filenameWE + "_after.json");

  return true;
}

} // namespace dawn
