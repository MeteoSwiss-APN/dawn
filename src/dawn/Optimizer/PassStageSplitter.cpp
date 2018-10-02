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
#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/ReadBeforeWriteConflict.h"
#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/AST.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/Logging.h"
#include <deque>
#include <iostream>
#include <iterator>
#include <unordered_set>

namespace dawn {

PassStageSplitter::PassStageSplitter() : Pass("PassStageSplitter", true) {}

bool PassStageSplitter::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {

  int numSplit = 0;
  std::deque<int> splitterIndices;
  std::deque<std::shared_ptr<iir::DependencyGraphAccesses>> graphs;

  // Iterate over all stages in all multistages of all stencils
  for(const auto& stencil : stencilInstantiation->getIIR()->getChildren()) {

    int multiStageIndex = 0;
    int linearStageIndex = 0;
    for(const auto& multiStage : stencil->getChildren()) {

      int stageIndex = 0;
      for(auto stageIt = multiStage->childrenBegin(); stageIt != multiStage->childrenEnd();
          ++stageIndex, ++linearStageIndex) {
        iir::Stage& stage = (**stageIt);
        iir::DoMethod& doMethod = stage.getSingleDoMethod();

        splitterIndices.clear();
        graphs.clear();

        std::shared_ptr<iir::DependencyGraphAccesses> newGraph, oldGraph;
        newGraph = std::make_shared<iir::DependencyGraphAccesses>(stencilInstantiation->getIIR().get());

        // Build the Dependency graph (bottom to top)
        for(int stmtIndex = doMethod.getChildren().size() - 1; stmtIndex >= 0; --stmtIndex) {
          auto& stmtAccessesPair = doMethod.getChildren()[stmtIndex];

          newGraph->insertStatementAccessesPair(stmtAccessesPair);

          // If we have a horizontal read-before-write conflict, we record the current index for
          // splitting
          if(hasHorizontalReadBeforeWriteConflict(newGraph.get())) {

            if(stencilInstantiation->getIIR()->getOptions().DumpSplitGraphs)
              oldGraph->toDot(
                  format("stmt_hd_ms%i_s%i_%02i.dot", multiStageIndex, stageIndex, numSplit));

            // Set the splitter index and assign the *old* graph as the stage dependency graph
            splitterIndices.push_front(stmtIndex);
            graphs.push_front(std::move(oldGraph));

            if(stencilInstantiation->getIIR()->getOptions().ReportPassStageSplit)
              std::cout << "\nPASS: " << getName() << ": " << stencilInstantiation->getIIR()->getMetaData()->getName()
                        << ": split:"
                        << stmtAccessesPair->getStatement()->ASTStmt->getSourceLocation().Line
                        << "\n";

            // Clear the new graph an process the current statements again
            newGraph->clear();
            newGraph->insertStatementAccessesPair(stmtAccessesPair);

            numSplit++;
          }

          oldGraph = newGraph->clone();
        }

        if(stencilInstantiation->getIIR()->getOptions().DumpSplitGraphs)
          newGraph->toDot(
              format("stmt_hd_ms%i_s%i_%02i.dot", multiStageIndex, stageIndex, numSplit));

        graphs.push_front(newGraph);

        // Perform the spliting of the stages and insert the stages *before* the stage we processed.
        // Note that the "old" stage will be erased (it was consumed in split(...) anyway)
        if(!splitterIndices.empty()) {
          auto newStages = stage.split(splitterIndices, &graphs);
          stageIt = multiStage->childrenErase(stageIt);
          multiStage->insertChildren(stageIt, std::make_move_iterator(newStages.begin()),
                                     std::make_move_iterator(newStages.end()));
        } else {
          DAWN_ASSERT(graphs.size() == 1);
          doMethod.setDependencyGraph(graphs.back());
          stage.update(iir::NodeUpdateType::level);
          ++stageIt;
        }
      }

      multiStageIndex += 1;
    }

    for(const auto& multiStage : stencil->getChildren()) {
      multiStage->update(iir::NodeUpdateType::levelAndTreeAbove);
    }
  }

  if(stencilInstantiation->getIIR()->getOptions().ReportPassStageSplit && !numSplit)
    std::cout << "\nPASS: " << getName() << ": " << stencilInstantiation->getIIR()->getMetaData()->getName()
              << ": no split\n";

  return true;
}

} // namespace dawn
