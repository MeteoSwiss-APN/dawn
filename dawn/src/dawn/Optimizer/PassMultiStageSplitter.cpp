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
#include "dawn/IIR/AST.h"
#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/ReadBeforeWriteConflict.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/Logger.h"

#include <deque>
#include <iterator>
#include <unordered_set>

namespace dawn {

typedef std::function<void(
    iir::MultiStage::child_reverse_iterator_t&, iir::DependencyGraphAccesses&, iir::LoopOrderKind&,
    iir::LoopOrderKind&, std::deque<iir::MultiStage::SplitIndex>&, int, int, int&,
    const std::string&, const std::string&, const std::string&, const Options&)>
    SplitterFunction;

namespace {

int checkDependencies(const std::unique_ptr<iir::Stage>& stage, int stmtIdx) {
  DAWN_ASSERT_MSG(false, "implementation missing");
  return 0;
}

SplitterFunction multiStageSplitterOptimized() {
  return [&](iir::MultiStage::child_reverse_iterator_t& stageIt,
             iir::DependencyGraphAccesses& graph, iir::LoopOrderKind userSpecifiedLoopOrder,
             iir::LoopOrderKind& curLoopOrder,
             std::deque<iir::MultiStage::SplitIndex>& splitterIndices, int stageIndex,
             int multiStageIndex, int& numSplit, const std::string& fileName,
             const std::string& StencilName, const std::string& PassName, const Options& options) {
    iir::Stage& stage = (**stageIt);
    iir::DoMethod& doMethod = stage.getSingleDoMethod();

    // Iterate statements backwards
    for(int stmtIndex = doMethod.getAST().getStatements().size() - 1; stmtIndex >= 0; --stmtIndex) {
      const auto& stmt = doMethod.getAST().getStatements()[stmtIndex];
      graph.insertStatement(stmt);

      // Check for read-before-write conflicts in the loop order and counter loop order.
      // Conflicts in the loop order will assure us that the multi-stage can't be
      // parallel. A conflict in the counter loop order is more severe and needs the current
      // multistage be split!
      auto conflict = hasVerticalReadBeforeWriteConflict(graph, userSpecifiedLoopOrder);
      if(conflict.CounterLoopOrderConflict) {

        // The loop order of the lower part is what we recoreded in the last steps
        splitterIndices.emplace_front(
            iir::MultiStage::SplitIndex{stageIndex, stmtIndex, curLoopOrder});

        DAWN_DIAG(INFO, fileName, doMethod.getAST().getStatements()[stmtIndex]->getSourceLocation())
            << StencilName << ": split, looporder: " << curLoopOrder;

        if(options.DumpSplitGraphs)
          graph.toDot(format("stmt_vd_ms%i_%02i.dot", multiStageIndex, numSplit));

        // Clear the graph ...
        graph.clear();
        curLoopOrder = iir::LoopOrderKind::Parallel;

        // ... and process the current statement again
        graph.insertStatement(stmt);

        numSplit++;

      } else if(conflict.LoopOrderConflict)
        // We have a conflict in the loop order, the multi-stage cannot be executed in
        // parallel and we use the loop order specified by the user
        curLoopOrder = userSpecifiedLoopOrder;
    }
  };
}

SplitterFunction multiStageSplitterDebug() {
  return [&](iir::MultiStage::child_reverse_iterator_t& stageIt,
             iir::DependencyGraphAccesses& graph, iir::LoopOrderKind& userSpecifiedLoopOrder,
             iir::LoopOrderKind& curLoopOrder,
             std::deque<iir::MultiStage::SplitIndex>& splitterIndices, int stageIndex,
             int multiStageIndex, int& numSplit, const std::string& fileName,
             const std::string& StencilName, const std::string& PassName, const Options& options) {
    DAWN_ASSERT_MSG(false, "Max-Cut for Multistages is not yet implemented");
    iir::Stage& stage = (**stageIt);
    iir::DoMethod& doMethod = stage.getSingleDoMethod();

    //==============================================================================================
    // Max-Cut dependency analysis missing
    //==============================================================================================
    // The missing piece of the max-cut algorithm still requires one check that does not cut
    // multistages that have loop order dependendices is the form of:
    //
    // vertical_region(k_start, k_end){
    //    c = a[k-1] + b;
    //    a += c;
    // }
    //
    // This pattern, an iterative solver as we call it, has vertical dependencies that need to
    // be detected and the statements need to be linked. The splitting then splits only if no
    // links to other statements are found.
    // Iterate statements backwards
    int openDependencies = 0;
    for(int stmtIndex = doMethod.getAST().getStatements().size() - 2; stmtIndex >= 0; --stmtIndex) {
      openDependencies += checkDependencies(*stageIt, stmtIndex);
      if(openDependencies == 0) {
        splitterIndices.emplace_front(
            iir::MultiStage::SplitIndex{stageIndex, stmtIndex, curLoopOrder});
        numSplit++;
      }
    }

    // After splitting all the statements into their own Multistages, we need to ensure that if
    // we have loop-order conflicts in one of those, that we resolve them properly.
    // For example if we find a multistage with an offset read/write pattern (a = b * a[k-1]),
    // even though this statement is independent from all other statements, we still need to
    // ensure the proper loop order and cannot just assume parallel.
    for(int stmtIndex = doMethod.getAST().getStatements().size() - 1; stmtIndex >= 0; --stmtIndex) {
      const auto& stmt = doMethod.getAST().getStatements()[stmtIndex];
      graph.insertStatement(stmt);

      // Check for read-before-write conflicts in the loop order.
      auto conflict = hasVerticalReadBeforeWriteConflict(graph, userSpecifiedLoopOrder);
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
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const Options& options) {

  SplitterFunction multistagesplitter = (strategy_ == MultiStageSplittingStrategy::Optimized)
                                            ? multiStageSplitterOptimized()
                                            : multiStageSplitterDebug();

  iir::DependencyGraphAccesses graph(stencilInstantiation->getMetaData());
  int numSplit = 0;
  std::string fileName = stencilInstantiation->getMetaData().getFileName();
  std::string StencilName = stencilInstantiation->getName();
  std::string PassName = getName();

  // Iterate over all stages in all multistages of all stencils
  for(const auto& stencil : stencilInstantiation->getStencils()) {
    int multiStageIndex = 0;

    for(auto multiStageIt = stencil->childrenBegin(); multiStageIt != stencil->childrenEnd();
        ++multiStageIndex) {
      iir::MultiStage& multiStage = (**multiStageIt);

      std::deque<iir::MultiStage::SplitIndex> splitterIndices;
      graph.clear();

      // Loop order specified by the user in the vertical region (e.g {k_start, k_end} is
      // forward)
      auto userSpecifiedLoopOrder = multiStage.getLoopOrder();

      // If not proven otherwise, we assume a parralel loop order
      auto curLoopOrder = iir::LoopOrderKind::Parallel;

      // Build the Dependency graph (bottom --> top i.e iterate the stages backwards)
      int stageIndex = multiStage.getChildren().size() - 1;
      for(auto stageIt = multiStage.childrenRBegin(); stageIt != multiStage.childrenREnd();
          ++stageIt, --stageIndex) {

        multistagesplitter(stageIt, graph, userSpecifiedLoopOrder, curLoopOrder, splitterIndices,
                           stageIndex, multiStageIndex, numSplit, fileName, StencilName, PassName,
                           options);
      }
      if(options.DumpSplitGraphs)
        graph.toDot(format("stmt_vd_m%i_%02i.dot", multiStageIndex, numSplit));

      if(!splitterIndices.empty()) {
        auto newMultiStages = multiStage.split(splitterIndices, curLoopOrder);
        multiStageIt = stencil->childrenErase(multiStageIt);
        stencil->insertChildren(multiStageIt, std::make_move_iterator(newMultiStages.begin()),
                                std::make_move_iterator(newMultiStages.end()));
      } else {
        ++multiStageIt;
        multiStage.setLoopOrder(curLoopOrder);
      }
    }
  }

  if(!numSplit)
    DAWN_LOG(INFO) << stencilInstantiation->getName() << ": no split";

  return true;
}

} // namespace dawn
