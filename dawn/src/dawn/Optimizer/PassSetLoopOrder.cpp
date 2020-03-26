#include "dawn/Optimizer/PassSetLoopOrder.h"
#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/IIR/DoMethod.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/LoopOrder.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/ReadBeforeWriteConflict.h"

namespace dawn {
PassSetLoopOrder::PassSetLoopOrder(OptimizerContext& context) : Pass(context, "PassSetLoopOrder") {}

bool PassSetLoopOrder::run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  iir::DependencyGraphAccesses graph(stencilInstantiation->getMetaData());
  for(auto& multiStage : iterateIIROver<iir::MultiStage>(*(stencilInstantiation->getIIR()))) {
    // analysis is on a multistage level, clear the graph for each new one
    graph.clear();
    auto userSpecifiedLoopOrder = multiStage->getLoopOrder();
    // try for a parallel loop order. This will be reverted if we run into a conflict
    multiStage->setLoopOrder(iir::LoopOrderKind::Parallel);
    for(auto& doMethod : iterateIIROver<iir::DoMethod>(*(multiStage))) {
      // Iterate statements backwards
      for(int stmtIndex = doMethod->getAST().getStatements().size() - 1; stmtIndex >= 0;
          --stmtIndex) {
        const auto& stmt = doMethod->getAST().getStatements()[stmtIndex];
        graph.insertStatement(stmt);
        // Check for read-before-write conflicts in the loop order and counter loop order.
        // Conflicts will assure us that the multi-stage can't be executed in  parallel.
        auto conflict = hasVerticalReadBeforeWriteConflict(graph, userSpecifiedLoopOrder);
        if(conflict.CounterLoopOrderConflict || conflict.LoopOrderConflict) {
          multiStage->setLoopOrder(userSpecifiedLoopOrder);
          break;
        }
      }
    }
  }
  return true;
}

} // namespace dawn