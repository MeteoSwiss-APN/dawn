//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gsl/Optimizer/PassStageReordering.h"
#include "gsl/Optimizer/OptimizerContext.h"
#include "gsl/Optimizer/StencilInstantiation.h"
#include "gsl/Support/FileUtil.h"
#include "gsl/Support/Unreachable.h"

#include "gsl/Optimizer/ReorderStrategyGreedy.h"
#include "gsl/Optimizer/ReorderStrategyPartitioning.h"

namespace gsl {

PassStageReordering::PassStageReordering(ReorderStrategy::ReorderStrategyKind strategy)
    : Pass("PassStageReordering"), strategy_(strategy) {
  dependencies_.push_back("PassSetStageGraph");
}

bool PassStageReordering::run(StencilInstantiation* stencilInstantiation) {
  OptimizerContext* context = stencilInstantiation->getOptimizerContext();

  std::string filenameWE = getFilenameWithoutExtension(context->getSIR()->Filename);
  if(context->getOptions().ReportPassStageReodering)
    stencilInstantiation->dumpAsJson(filenameWE + "_before.json", getName());

  for(auto& stencilPtr : stencilInstantiation->getStencils()) {
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
      gsl_unreachable("invalid reorder strategy");
    }

    stencilPtr = strategy->reorder(stencilPtr);
    if(!stencilPtr)
      return false;
  }

  if(context->getOptions().ReportPassStageReodering)
    stencilInstantiation->dumpAsJson(filenameWE + "_after.json", getName());

  return true;
}

} // namespace gsl
