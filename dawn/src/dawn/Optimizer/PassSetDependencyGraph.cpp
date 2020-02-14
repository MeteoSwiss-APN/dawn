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

#include "dawn/Optimizer/PassSetDependencyGraph.h"
#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/IIR/StencilInstantiation.h"

namespace dawn {

PassSetDependencyGraph::PassSetDependencyGraph(OptimizerContext& context)
    : Pass(context, "PassSetDependencyGraph") {}

bool PassSetDependencyGraph::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  const auto& IIR = stencilInstantiation->getIIR();
  for(const auto& doMethods : iterateIIROver<iir::DoMethod>(*IIR)) {
    // and do the update of the Graphs
    doMethods->update(iir::NodeUpdateType::levelAndTreeAbove);
    std::shared_ptr<iir::DependencyGraphAccesses> newGraph;
    newGraph = std::make_shared<iir::DependencyGraphAccesses>(stencilInstantiation->getMetaData());
    // Build the Dependency graph (bottom to top)
    for(int stmtIndex = doMethods->getAST().getStatements().size() - 1; stmtIndex >= 0;
        --stmtIndex) {
      const auto& stmt = doMethods->getAST().getStatements()[stmtIndex];

      newGraph->insertStatement(stmt);
      doMethods->setDependencyGraph(newGraph);
    }
    // and do the update
    doMethods->update(iir::NodeUpdateType::levelAndTreeAbove);
  }
  return true;
}
} // namespace dawn