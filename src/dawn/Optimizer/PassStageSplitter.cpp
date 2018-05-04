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

#include "dawn/Optimizer/PassStageSplitter.h"
#include "dawn/Optimizer/DependencyGraphAccesses.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/ReadBeforeWriteConflict.h"
#include "dawn/Optimizer/StatementAccessesPair.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/SIR/AST.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/Logging.h"
#include <deque>
#include <iostream>
#include <iterator>
#include <unordered_set>

namespace dawn {

PassStageSplitter::PassStageSplitter(StageSplittingStrategy strategy)
    : Pass("PassStageSplitter"), strategy_(strategy) {
  isDebug_ = true;
}

namespace {
std::function<void(std::list<std::shared_ptr<Stage>>::iterator&, std::deque<int>&,
                   std::deque<std::shared_ptr<DependencyGraphAccesses>>&,
                   const std::shared_ptr<StencilInstantiation>&, const Options&, int, int,
                   const std::string&, const std::string&, int&, std::shared_ptr<MultiStage>&)>
setOptimizedLoopContent() {
  return [&](std::list<std::shared_ptr<Stage>>::iterator& stageIt, std::deque<int>& splitterIndices,
             std::deque<std::shared_ptr<DependencyGraphAccesses>>& graphs,
             const std::shared_ptr<StencilInstantiation>& stencilInstantiation,
             const Options& options, int multiStageIndex, int stageIndex,
             const std::string& passName, const std::string& stencilName, int& numSplit,
             std::shared_ptr<MultiStage>& multiStage) {

    Stage& stage = (**stageIt);
    DoMethod& doMethod = stage.getSingleDoMethod();

    splitterIndices.clear();
    graphs.clear();

    std::shared_ptr<DependencyGraphAccesses> newGraph, oldGraph;
    newGraph = std::make_shared<DependencyGraphAccesses>(stencilInstantiation.get());

    // Build the Dependency graph (bottom to top)
    for(int stmtIndex = doMethod.getStatementAccessesPairs().size() - 1; stmtIndex >= 0;
        --stmtIndex) {
      auto& stmtAccessesPair = doMethod.getStatementAccessesPairs()[stmtIndex];

      newGraph->insertStatementAccessesPair(stmtAccessesPair);

      // If we have a horizontal read-before-write conflict, we record the current index for
      // splitting
      if(hasHorizontalReadBeforeWriteConflict(newGraph.get())) {

        if(options.DumpSplitGraphs)
          oldGraph->toDot(
              format("stmt_hd_ms%i_s%i_%02i.dot", multiStageIndex, stageIndex, numSplit));

        // Set the splitter index and assign the *old* graph as the stage dependency graph
        splitterIndices.push_front(stmtIndex);
        graphs.push_front(std::move(oldGraph));

        if(options.ReportPassStageSplit)
          std::cout << "\nPASS: " << passName /*getName()*/ << ": "
                    << stencilName /*stencilInstantiation->getName()*/
                    << ": split:"
                    << stmtAccessesPair->getStatement()->ASTStmt->getSourceLocation().Line << "\n";

        // Clear the new graph an process the current statements again
        newGraph->clear();
        newGraph->insertStatementAccessesPair(stmtAccessesPair);

        numSplit++;
      }

      oldGraph = newGraph->clone();
    }

    if(options.DumpSplitGraphs)
      newGraph->toDot(format("stmt_hd_ms%i_s%i_%02i.dot", multiStageIndex, stageIndex, numSplit));

    graphs.push_front(newGraph);

    // Perform the spliting of the stages and insert the stages *before* the stage we processed.
    // Note that the "old" stage will be erased (it was consumed in split(...) anyway)
    if(!splitterIndices.empty()) {
      auto newStages = stage.split(splitterIndices, &graphs);
      stageIt = multiStage->getStages().erase(stageIt);
      multiStage->getStages().insert(stageIt, std::make_move_iterator(newStages.begin()),
                                     std::make_move_iterator(newStages.end()));
    } else {
      DAWN_ASSERT(graphs.size() == 1);
      doMethod.setDependencyGraph(graphs.back());
      stage.update();
      ++stageIt;
    }
  };
}
std::function<void(std::list<std::shared_ptr<Stage>>::iterator&, std::deque<int>&,
                   std::deque<std::shared_ptr<DependencyGraphAccesses>>&,
                   const std::shared_ptr<StencilInstantiation>&, const Options&, int, int,
                   const std::string&, const std::string&, int&, std::shared_ptr<MultiStage>&)>
setDebugLoopContent() {

  return
      [&](std::list<std::shared_ptr<Stage>>::iterator& stageIt, std::deque<int>& splitterIndices,
          std::deque<std::shared_ptr<DependencyGraphAccesses>>& graphs,
          const std::shared_ptr<StencilInstantiation>& stencilInstantiation, const Options& options,
          int multiStageIndex, int stageIndex, const std::string& passName,
          const std::string& stencilName, int& numSplit, std::shared_ptr<MultiStage>& multiStage) {
        Stage& stage = (**stageIt);
        DoMethod& doMethod = stage.getSingleDoMethod();

        splitterIndices.clear();
        graphs.clear();

        std::shared_ptr<DependencyGraphAccesses> newGraph, oldGraph;
        newGraph = std::make_shared<DependencyGraphAccesses>(stencilInstantiation.get());

        // Build the Dependency graph (bottom to top)
        for(int stmtIndex = doMethod.getStatementAccessesPairs().size() - 2; stmtIndex >= 0;
            --stmtIndex) {
          auto& stmtAccessesPair = doMethod.getStatementAccessesPairs()[stmtIndex];

          newGraph->insertStatementAccessesPair(stmtAccessesPair);

          // Set the splitter index and assign the *old* graph as the stage dependency graph
          splitterIndices.push_front(stmtIndex);
          graphs.push_front(std::move(oldGraph));

          // Clear the new graph an process the current statements again
          newGraph->clear();
          newGraph->insertStatementAccessesPair(stmtAccessesPair);

          numSplit++;

          oldGraph = newGraph->clone();
        }

        graphs.push_front(newGraph);

        // Perform the spliting of the stages and insert the stages *before* the stage we processed.
        // Note that the "old" stage will be erased (it was consumed in split(...) anyway)
        if(!splitterIndices.empty()) {
          auto newStages = stage.split(splitterIndices, &graphs);
          stageIt = multiStage->getStages().erase(stageIt);
          multiStage->getStages().insert(stageIt, std::make_move_iterator(newStages.begin()),
                                         std::make_move_iterator(newStages.end()));
        } else {
          DAWN_ASSERT(graphs.size() == 1);
          doMethod.setDependencyGraph(graphs.back());
          stage.update();
          ++stageIt;
        }

      };
}
} // anonymous namespace

bool PassStageSplitter::run(const std::shared_ptr<StencilInstantiation>& stencilInstantiation) {
  OptimizerContext* context = stencilInstantiation->getOptimizerContext();

  int numSplit = 0;
  std::deque<int> splitterIndices;
  std::deque<std::shared_ptr<DependencyGraphAccesses>> graphs;

  std::function<void(std::list<std::shared_ptr<Stage>>::iterator&, std::deque<int>&,
                     std::deque<std::shared_ptr<DependencyGraphAccesses>>&,
                     const std::shared_ptr<StencilInstantiation>&, const Options&, int, int,
                     const std::string&, const std::string&, int&, std::shared_ptr<MultiStage>&)>
      stageSplittingAlgorithm;

  if(strategy_ == StageSplittingStrategy::SS_Optimized) {
    stageSplittingAlgorithm = setOptimizedLoopContent();
  } else {
    DAWN_ASSERT_MSG(
        false, "Debug splitting strategy for stages is not supported by the extent computation");
    stageSplittingAlgorithm = setDebugLoopContent();
  }

  // Iterate over all stages in all multistages of all stencils
  for(auto& stencil : stencilInstantiation->getStencils()) {

    int multiStageIndex = 0;
    int linearStageIndex = 0;
    for(auto& multiStage : stencil->getMultiStages()) {

      int stageIndex = 0;
      for(auto stageIt = multiStage->getStages().begin(); stageIt != multiStage->getStages().end();
          ++stageIndex, ++linearStageIndex) {
        stageSplittingAlgorithm(stageIt, splitterIndices, graphs, stencilInstantiation,
                                context->getOptions(), multiStageIndex, stageIndex, getName(),
                                stencilInstantiation->getName(), numSplit, multiStage);
      }

      multiStageIndex += 1;
    }
  }

  if(context->getOptions().ReportPassStageSplit && !numSplit)
    std::cout << "\nPASS: " << getName() << ": " << stencilInstantiation->getName()
              << ": no split\n";

  return true;
}

} // namespace dawn
