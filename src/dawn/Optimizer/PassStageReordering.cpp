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
#include "dawn/Support/FileUtil.h"
#include "dawn/Support/Unreachable.h"

#include "dawn/Optimizer/ReorderStrategyGreedy.h"
#include "dawn/Optimizer/ReorderStrategyPartitioning.h"
namespace dawn {

PassStageReordering::PassStageReordering(ReorderStrategy::ReorderStrategyKind strategy)
    : Pass("PassStageReordering"), strategy_(strategy) {
  dependencies_.push_back("PassSetStageGraph");
}

bool PassStageReordering::run(
    const std::unique_ptr<iir::IIR>& iir) {

  std::string filenameWE =
      getFilenameWithoutExtension(iir->getMetaData()->getFileName());
  if(iir->getOptions().ReportPassStageReodering) {
//        stencilInstantiation->dumpAsJson(filenameWE + "_before.json", getName());
    ///// Todo add the serializer here once this is merged
  }
  for(const auto& stencilPtr : iir->getChildren()) {
    if(strategy_ == ReorderStrategy::RK_None)
      continue;

    std::unique_ptr<ReorderStrategy> strategy;
    switch(strategy_) {
    case ReorderStrategy::RK_Greedy:
      strategy = make_unique<ReoderStrategyGreedy>();
      break;
    case ReorderStrategy::RK_Partitioning:
      strategy = make_unique<ReoderStrategyPartitioning>();
      break;
    default:
      dawn_unreachable("invalid reorder strategy");
    }

    // TODO should we have Iterators so to prevent unique_ptr swaps
    auto newStencil = strategy->reorder(stencilPtr);

    iir->replace(stencilPtr, newStencil, iir);

    stencilPtr->update(iir::NodeUpdateType::levelAndTreeAbove);

    if(!stencilPtr)
      return false;
  }

  if(iir->getOptions().ReportPassStageReodering) {
    //    stencilInstantiation->dumpAsJson(filenameWE + "_after.json", getName());
    /// Todo add the serializer here once this is merged
  }
  return true;
}

} // namespace dawn
