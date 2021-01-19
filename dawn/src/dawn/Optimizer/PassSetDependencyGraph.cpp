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

bool PassSetDependencyGraph::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const Options& options) {
  const auto& IIR = stencilInstantiation->getIIR();
  for(const auto& doMethod : iterateIIROver<iir::DoMethod>(*IIR)) {
    // and do the update of the Graphs
    doMethod->update(iir::NodeUpdateType::levelAndTreeAbove);
    iir::DependencyGraphAccesses newGraph(stencilInstantiation->getMetaData());
    // Build the Dependency graph (bottom to top)
    for(int stmtIndex = doMethod->getAST().getStatements().size() - 1; stmtIndex >= 0;
        --stmtIndex) {
      const auto& stmt = doMethod->getAST().getStatements()[stmtIndex];

      newGraph.insertStatement(stmt);
    }
    doMethod->setDependencyGraph(std::move(newGraph));
    // and do the update
    doMethod->update(iir::NodeUpdateType::levelAndTreeAbove);
  }
  return true;
}
} // namespace dawn
