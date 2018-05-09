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

PassMultiStageSplitter::PassMultiStageSplitter(MultiStageSplittingStrategy strategy)
    : Pass("PassMultiStageSplitter", true), strategy_(strategy) {}
namespace {

int checkDependencies(const std::shared_ptr<Stage>& stage, int stmtIdx) {
  DAWN_ASSERT_MSG(false, "implementation missing");
  return 0;
}

std::function<void(std::list<std::shared_ptr<Stage>>::reverse_iterator&, DependencyGraphAccesses&,
                   LoopOrderKind&, LoopOrderKind&, std::deque<MultiStage::SplitIndex>&, int, int,
                   int&, const std::string&, const std::string&, const Options&)>
multiStageSplitterOptimzied() {
  return [&](std::list<std::shared_ptr<Stage>>::reverse_iterator& stageIt,
             DependencyGraphAccesses& graph, LoopOrderKind userSpecifiedLoopOrder,
             LoopOrderKind& curLoopOrder, std::deque<MultiStage::SplitIndex>& splitterIndices,
             int stageIndex, int multiStageIndex, int& numSplit, const std::string& StencilName,
             const std::string& PassName, const Options& options) {

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
        splitterIndices.emplace_front(MultiStage::SplitIndex{stageIndex, stmtIndex, curLoopOrder});

        if(options.ReportPassMultiStageSplit)
          std::cout << "\nPASS: " << PassName << ": " << StencilName << ": split:"
                    << doMethod.getStatementAccessesPairs()[stmtIndex]
                           ->getStatement()
                           ->ASTStmt->getSourceLocation()
                           .Line
                    << " looporder:" << curLoopOrder << "\n";

        if(options.DumpSplitGraphs)
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
  };
}
std::function<void(std::list<std::shared_ptr<Stage>>::reverse_iterator&, DependencyGraphAccesses&,
                   LoopOrderKind&, LoopOrderKind&, std::deque<MultiStage::SplitIndex>&, int, int,
                   int&, const std::string&, const std::string&, const Options&)>
multiStageSplitterDebug() {

  return [&](std::list<std::shared_ptr<Stage>>::reverse_iterator& stageIt,
             DependencyGraphAccesses& graph, LoopOrderKind& userSpecifiedLoopOrder,
             LoopOrderKind& curLoopOrder, std::deque<MultiStage::SplitIndex>& splitterIndices,
             int stageIndex, int multiStageIndex, int& numSplit, const std::string& StencilName,
             const std::string& PassName, const Options& options) {
    DAWN_ASSERT_MSG(false, "Max-Cut for Multistages is not yet implemented");
    Stage& stage = (**stageIt);
    DoMethod& doMethod = stage.getSingleDoMethod();

    //==============================================================================================
    // Max-Cut dependency analysis missing
    //==============================================================================================
    // The missing piece of the max-cut algorithm still requires one check that does not cut
    // multistages that have loop order depdendencies is the form of:
    //
    // vertical_region(k_start, k_end){
    //    c = a[k-1] + b;
    //    a += c;
    // }
    //
    // This pattern, an iterative solver as we call it, has vertical dependencies that need to be
    // detected and the statements need to be linked. The splitting then splits only if no links to
    // other statements are found.

    // Iterate statements backwards
    int openDependencies = 0;
    for(int stmtIndex = doMethod.getStatementAccessesPairs().size() - 2; stmtIndex >= 0;
        --stmtIndex) {
      openDependencies += checkDependencies(*stageIt, stmtIndex);
      if(openDependencies == 0) {
        splitterIndices.emplace_front(MultiStage::SplitIndex{stageIndex, stmtIndex, curLoopOrder});
        numSplit++;
      }
    }

    // After splitting all the statements into their own Multistages, we need to ensure that if we
    // have loop-order conflicts in one of those, that we resolve them properly.
    //
    // For example if we find a multistage with an offset read/write pattern (a = b * a[k-1]), even
    // though this statement is independent from all other statements, we still need to ensure the
    // proper loop order and cannot just assume parallel.
    for(int stmtIndex = doMethod.getStatementAccessesPairs().size() - 1; stmtIndex >= 0;
        --stmtIndex) {
      auto& stmtAccessesPair = doMethod.getStatementAccessesPairs()[stmtIndex];
      graph.insertStatementAccessesPair(stmtAccessesPair);

      // Check for read-before-write conflicts in the loop order.
      auto conflict = hasVerticalReadBeforeWriteConflict(&graph, userSpecifiedLoopOrder);
      if(conflict.LoopOrderConflict) {
        // We have a conflict in the loop order, the multi-stage cannot be executed in
        // parallel and we use the loop order specified by the user
        curLoopOrder = userSpecifiedLoopOrder;
      }
    }
  };
}

} // namespace

bool PassMultiStageSplitter::run(
    const std::shared_ptr<StencilInstantiation>& stencilInstantiation) {

  std::function<void(std::list<std::shared_ptr<Stage>>::reverse_iterator&, DependencyGraphAccesses&,
                     LoopOrderKind&, LoopOrderKind&, std::deque<MultiStage::SplitIndex>&, int, int,
                     int&, const std::string&, const std::string&, const Options&)>
      multistagesplitter;

  if(strategy_ == MultiStageSplittingStrategy::SS_Optimized) {
    multistagesplitter = multiStageSplitterOptimzied();
  } else {
    multistagesplitter = multiStageSplitterDebug();
  }

  OptimizerContext* context = stencilInstantiation->getOptimizerContext();
  DependencyGraphAccesses graph(stencilInstantiation.get());
  int numSplit = 0;
  std::string StencilName = stencilInstantiation->getName();
  std::string PassName = getName();
  auto options = context->getOptions();

  // Iterate over all stages in all multistages of all stencils
  for(auto& stencil : stencilInstantiation->getStencils()) {
    int multiStageIndex = 0;

    for(auto multiStageIt = stencil->getMultiStages().begin();
        multiStageIt != stencil->getMultiStages().end(); ++multiStageIndex) {
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

        multistagesplitter(stageIt, graph, userSpecifiedLoopOrder, curLoopOrder, splitterIndices,
                           stageIndex, multiStageIndex, numSplit, StencilName, PassName, options);
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
        multiStage.setLoopOrder(curLoopOrder);
      }
    }
  }

  if(context->getOptions().ReportPassMultiStageSplit && !numSplit)
    std::cout << "\nPASS: " << getName() << ": " << stencilInstantiation->getName()
              << ": no split\n";

  return true;
}

} // namespace dawn
