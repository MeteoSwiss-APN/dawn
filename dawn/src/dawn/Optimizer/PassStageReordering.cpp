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
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Support/FileSystem.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {

bool PassStageReordering::run(
    const std::shared_ptr<iir::StencilInstantiation>& instantiation,
    const Options& options) {
  const std::string filenameWE =
      fs::path(instantiation->getMetaData().getFileName()).filename().stem();

  if(options.WriteStencilInstantiation)
    instantiation->jsonDump(filenameWE + "_before_stage_reordering.json");

  for(const auto& stencil : instantiation->getStencils()) {
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
    auto newStencil = strategy->reorder(instantiation.get(), stencil, options);
    if(!newStencil)
      return false;

    instantiation->getIIR()->replace(stencil, newStencil, instantiation->getIIR());
    newStencil->update(iir::NodeUpdateType::levelAndTreeAbove);
  }

  if(options.WriteStencilInstantiation)
    instantiation->jsonDump(filenameWE + "_after_stage_reordering.json");

  return true;
}

} // namespace dawn
