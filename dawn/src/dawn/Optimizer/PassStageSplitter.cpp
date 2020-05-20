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
#include "dawn/IIR/AST.h"
#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/ReadBeforeWriteConflict.h"
#include "dawn/Support/Exception.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/Logger.h"

#include <deque>
#include <iterator>
#include <unordered_set>

namespace dawn {

PassStageSplitter::PassStageSplitter(OptimizerContext& context)
    : Pass(context, "PassStageSplitter", true) {}

bool PassStageSplitter::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  int numSplit = 0;

  // Iterate over all stages in all multistages of all stencils
  for(const auto& stencil : stencilInstantiation->getStencils()) {

    int multiStageIndex = 0;
    int linearStageIndex = 0;
    for(const auto& multiStage : stencil->getChildren()) {

      int stageIndex = 0;
      for(auto stageIt = multiStage->childrenBegin(); stageIt != multiStage->childrenEnd();
          ++stageIndex, ++linearStageIndex) {
        iir::Stage& stage = (**stageIt);
        iir::DoMethod& doMethod = stage.getSingleDoMethod();

        std::deque<int> splitterIndices;
        std::deque<iir::DependencyGraphAccesses> graphs;

        iir::DependencyGraphAccesses newGraph(stencilInstantiation->getMetaData());
        auto oldGraph = newGraph;

        // Build the Dependency graph (bottom to top)
        for(int stmtIndex = doMethod.getAST().getStatements().size() - 1; stmtIndex >= 0;
            --stmtIndex) {
          const auto& stmt = doMethod.getAST().getStatements()[stmtIndex];

          newGraph.insertStatement(stmt);

          // If we have a horizontal read-before-write conflict, we record the current index for
          // splitting
          if(hasHorizontalReadBeforeWriteConflict(newGraph)) {

            // Check if the conflict is related to a conditional block
            if(isa<iir::IfStmt>(stmt.get())) {
              // Check if the conflict is inside the conditional block
              iir::DependencyGraphAccesses conditionalBlockGraph =
                  iir::DependencyGraphAccesses(stencilInstantiation->getMetaData());
              conditionalBlockGraph.insertStatement(stmt);
              if(hasHorizontalReadBeforeWriteConflict(conditionalBlockGraph)) {
                // Since splitting inside a conditional block is not supported, report and return an
                // error.
                throw SyntacticError(
                    "Read-before-Write conflict inside conditional block is not supported.",
                    stencilInstantiation->getMetaData().getFileName(), stmt->getSourceLocation());
              }
            }

            if(context_.getOptions().DumpSplitGraphs)
              oldGraph.toDot(
                  format("stmt_hd_ms%i_s%i_%02i.dot", multiStageIndex, stageIndex, numSplit));

            // Set the splitter index and assign the *old* graph as the stage dependency graph
            splitterIndices.push_front(stmtIndex);
            graphs.push_front(std::move(oldGraph));

            DAWN_DIAG(INFO, stencilInstantiation->getMetaData().getFileName(),
                      stmt->getSourceLocation())
                << stencilInstantiation->getName() << ": split stage";

            // Clear the new graph an process the current statements again
            newGraph.clear();
            newGraph.insertStatement(stmt);

            numSplit++;
          }

          oldGraph = newGraph;
        }

        if(context_.getOptions().DumpSplitGraphs)
          newGraph.toDot(
              format("stmt_hd_ms%i_s%i_%02i.dot", multiStageIndex, stageIndex, numSplit));

        graphs.push_front(newGraph);

        // Perform the spliting of the stages and insert the stages *before* the stage we processed.
        // Note that the "old" stage will be erased (it was consumed in split(...) anyway)
        if(!splitterIndices.empty()) {
          auto newStages = stage.split(splitterIndices, std::move(graphs));
          stageIt = multiStage->childrenErase(stageIt);
          multiStage->insertChildren(stageIt, std::make_move_iterator(newStages.begin()),
                                     std::make_move_iterator(newStages.end()));
        } else {
          DAWN_ASSERT(graphs.size() == 1);
          doMethod.setDependencyGraph(std::move(graphs.back()));
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

  if(!numSplit)
    DAWN_LOG(INFO) << stencilInstantiation->getName() << ": no split";

  return true;
}

} // namespace dawn
