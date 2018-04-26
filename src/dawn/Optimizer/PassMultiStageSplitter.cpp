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

#include "dawn/Optimizer/PassMultiStageSplitter.h"
#include "dawn/Optimizer/DependencyGraphAccesses.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/ReadBeforeWriteConflict.h"
#include "dawn/Optimizer/StatementAccessesPair.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/SIR/AST.h"
#include "dawn/Support/Format.h"
#include <deque>
#include <iostream>
#include <iterator>
#include <unordered_set>

namespace dawn {

PassMultiStageSplitter::PassMultiStageSplitter(MulitStageSplittingStrategy strategy)
    : Pass("PassMultiStageSplitter"), strategy_(strategy) {
  isDebug_ = true;
}
namespace {
std::function<void(std::list<std::shared_ptr<MultiStage>>::iterator&, OptimizerContext*, int&,
                   DependencyGraphAccesses&, std::shared_ptr<Stencil>, int&, std::string&)>
setOptimizedLoopContent(const std::shared_ptr<StencilInstantiation>& stencilInstantiation) {
  return [&](std::list<std::shared_ptr<MultiStage>>::iterator& multiStageIt,
             OptimizerContext* context, int& numSplit, DependencyGraphAccesses& graph,
             std::shared_ptr<Stencil> stencil, int& multiStageIndex, std::string& name) {
    MultiStage& multiStage = (**multiStageIt);

    std::deque<MultiStage::SplitIndex> splitterIndices;
    graph.clear();

    // Loop order specified by the user in the vertical region (e.g {k_start, k_end} is
    // forward)
    auto userSpecifiedLoopOrder = multiStage.getLoopOrder();

    // If not proven otherwise, we assume a parralel loop order
    auto curLoopOrder = LoopOrderKind::LK_Parallel;

    // Build the Dependency graph (bottom --> top i.e iterate the stages backwards)
    int stageIndex = multiStage.getStages().size() - 1;
    for(auto stageIt = multiStage.getStages().rbegin(); stageIt != multiStage.getStages().rend();
        ++stageIt, --stageIndex) {
      Stage& stage = (**stageIt);
      DoMethod& doMethod = stage.getSingleDoMethod();

      // Iterate statements backwards
      for(int stmtIndex = doMethod.getStatementAccessesPairs().size() - 1; stmtIndex >= 0;
          --stmtIndex) {
        auto& stmtAccessesPair = doMethod.getStatementAccessesPairs()[stmtIndex];
        graph.insertStatementAccessesPair(stmtAccessesPair);

        // Check for read-before-write conflicts in the loop order and counter loop order.
        // Conflicts in the loop order will assure us that the multi-stage can't be
        // parallel. A conflict in the counter loop order is more severe and needs the current
        // multistage be splitted!
        auto conflict = hasVerticalReadBeforeWriteConflict(&graph, userSpecifiedLoopOrder);

        if(conflict.CounterLoopOrderConflict) {

          // The loop order of the lower part is what we recoreded in the last steps
          splitterIndices.emplace_front(
              MultiStage::SplitIndex{stageIndex, stmtIndex, curLoopOrder});

          if(context->getOptions().ReportPassMultiStageSplit)
            std::cout << "\nPASS: " << name << ": " << stencilInstantiation->getName() << ": split:"
                      << doMethod.getStatementAccessesPairs()[stmtIndex]
                             ->getStatement()
                             ->ASTStmt->getSourceLocation()
                             .Line
                      << " looporder:" << curLoopOrder << "\n";

          if(context->getOptions().DumpSplitGraphs)
            graph.toDot(format("stmt_vd_ms%i_%02i.dot", multiStageIndex, numSplit));

          // Clear the graph ...
          graph.clear();
          curLoopOrder = LoopOrderKind::LK_Parallel;

          // ... and process the current statement again
          graph.insertStatementAccessesPair(stmtAccessesPair);

          numSplit++;

        } else if(conflict.LoopOrderConflict)
          // We have a conflict in the loop order, the multi-stage cannot be executed in
          // parallel and we use the loop order specified by the user
          curLoopOrder = userSpecifiedLoopOrder;
      }
    }

    if(context->getOptions().DumpSplitGraphs)
      graph.toDot(format("stmt_vd_m%i_%02i.dot", multiStageIndex, numSplit));

    if(!splitterIndices.empty()) {
      auto newMultiStages = multiStage.split(splitterIndices, curLoopOrder);
      multiStageIt = stencil->getMultiStages().erase(multiStageIt);
      stencil->getMultiStages().insert(multiStageIt,
                                       std::make_move_iterator(newMultiStages.begin()),
                                       std::make_move_iterator(newMultiStages.end()));
    } else {
      ++multiStageIt;

      // Not split needed, however we still may have detected forward or backward loop order
      multiStage.setLoopOrder(curLoopOrder);
    }
  };
}
std::function<void(std::list<std::shared_ptr<MultiStage>>::iterator&, OptimizerContext*, int&,
                   DependencyGraphAccesses&, std::shared_ptr<Stencil>, int&, std::string&)>
setDebugLoopContent() {

  return [&](std::list<std::shared_ptr<MultiStage>>::iterator& multiStageIt,
             OptimizerContext* context, int& numSplit, DependencyGraphAccesses& graph,
             std::shared_ptr<Stencil> stencil, int& multiStageIndex, std::string& name) {
    MultiStage& multiStage = (**multiStageIt);

    std::deque<MultiStage::SplitIndex> splitterIndices;

    // Since every Statement has it's own loop order, we can assume parallel for all
    auto curLoopOrder = LoopOrderKind::LK_Parallel;

    int stageIndex = multiStage.getStages().size() - 1;
    for(auto stageIt = multiStage.getStages().rbegin(); stageIt != multiStage.getStages().rend();
        ++stageIt, --stageIndex) {
      Stage& stage = (**stageIt);
      DoMethod& doMethod = stage.getSingleDoMethod();

      // Iterate statements backwards
      for(int stmtIndex = doMethod.getStatementAccessesPairs().size() - 2; stmtIndex >= 0;
          --stmtIndex) {
        // We split every StmtAccessPair into its own multistage
        splitterIndices.emplace_front(MultiStage::SplitIndex{stageIndex, stmtIndex, curLoopOrder});
        numSplit++;
      }
    }

    if(!splitterIndices.empty()) {
      auto newMultiStages = multiStage.split(splitterIndices, curLoopOrder);
      multiStageIt = stencil->getMultiStages().erase(multiStageIt);
      stencil->getMultiStages().insert(multiStageIt,
                                       std::make_move_iterator(newMultiStages.begin()),
                                       std::make_move_iterator(newMultiStages.end()));
    } else {
      ++multiStageIt;
      multiStage.setLoopOrder(curLoopOrder);
    }
  };
}

} // namespace


bool PassMultiStageSplitter::run(
    const std::shared_ptr<StencilInstantiation>& stencilInstantiation) {

  std::function<void(std::list<std::shared_ptr<MultiStage>>::iterator&, OptimizerContext*, int&,
                     DependencyGraphAccesses&, std::shared_ptr<Stencil>, int&, std::string&)>
      multistagesplitter;

  if(strategy_ == MulitStageSplittingStrategy::SS_Optimized) {
    multistagesplitter = setOptimizedLoopContent(stencilInstantiation);
  } else {
    multistagesplitter = setDebugLoopContent();
  }

  OptimizerContext* context = stencilInstantiation->getOptimizerContext();
  DependencyGraphAccesses graph(stencilInstantiation.get());
  int numSplit = 0;
  std::string name = getName();

  // Iterate over all stages in all multistages of all stencils
  for(auto& stencil : stencilInstantiation->getStencils()) {
    int multiStageIndex = 0;

    for(auto multiStageIt = stencil->getMultiStages().begin();
        multiStageIt != stencil->getMultiStages().end(); ++multiStageIndex) {

      multistagesplitter(multiStageIt, context, numSplit, graph, stencil, multiStageIndex, name);
    }
  }

  if(context->getOptions().ReportPassMultiStageSplit && !numSplit)
    std::cout << "\nPASS: " << getName() << ": " << stencilInstantiation->getName()
              << ": no split\n";

  return true;
}

} // namespace dawn
