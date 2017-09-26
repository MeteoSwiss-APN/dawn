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

#include "gsl/Optimizer/PassMultiStageSplitter.h"
#include "gsl/Optimizer/DependencyGraphAccesses.h"
#include "gsl/Optimizer/OptimizerContext.h"
#include "gsl/Optimizer/ReadBeforeWriteConflict.h"
#include "gsl/Optimizer/StatementAccessesPair.h"
#include "gsl/Optimizer/StencilInstantiation.h"
#include "gsl/SIR/AST.h"
#include "gsl/Support/Format.h"
#include <deque>
#include <iostream>
#include <iterator>
#include <unordered_set>

namespace gsl {

PassMultiStageSplitter::PassMultiStageSplitter() : Pass("PassMultiStageSplitter") {}

bool PassMultiStageSplitter::run(StencilInstantiation* stencilInstantiation) {
  OptimizerContext* context = stencilInstantiation->getOptimizerContext();

  int numSplit = 0;
  std::deque<MultiStage::SplitIndex> splitterIndices;
  DependencyGraphAccesses graph(stencilInstantiation);

  // Iterate over all stages in all multistages of all stencils
  for(auto& stencil : stencilInstantiation->getStencils()) {

    int multiStageIndex = 0;
    for(auto multiStageIt = stencil->getMultiStages().begin();
        multiStageIt != stencil->getMultiStages().end(); ++multiStageIndex) {
      MultiStage& multiStage = (**multiStageIt);

      splitterIndices.clear();
      graph.clear();

      // Loop order specified by the user in the vertical region (e.g {k_start, k_end} is forward)
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
              std::cout << "\nPASS: " << getName() << ": " << stencilInstantiation->getName()
                        << ": split:"
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
            // We have a conflict in the loop order, the multi-stage cannot be executed in parallel
            // and we use the loop order specified by the user
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
    }
  }

  if(context->getOptions().ReportPassMultiStageSplit && !numSplit)
    std::cout << "\nPASS: " << getName() << ": " << stencilInstantiation->getName()
              << ": no split\n";

  return true;
}

} // namespace gsl
