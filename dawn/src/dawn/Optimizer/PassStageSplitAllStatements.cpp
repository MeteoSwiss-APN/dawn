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

#include "PassStageSplitAllStatements.h"
#include "dawn/IIR/DoMethod.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/Stage.h"
#include "dawn/IIR/StencilInstantiation.h"
#include <deque>
#include <iterator>

namespace dawn {

bool PassStageSplitAllStatements::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const Options& options) {
  for(const auto& multiStage : iterateIIROver<iir::MultiStage>(*stencilInstantiation->getIIR())) {
    for(auto stageIt = multiStage->childrenBegin(); stageIt != multiStage->childrenEnd();
        ++stageIt) {
      iir::Stage& stage = (**stageIt);
      iir::DoMethod& doMethod = stage.getSingleDoMethod();
      if(doMethod.getAST().getStatements().size() == 0) {
        continue;
      }
      std::deque<int> splitterIndices(doMethod.getAST().getStatements().size() - 1);
      // Fill splitterIndices with indices from 0 to number of statements - 1
      std::iota(splitterIndices.begin(), splitterIndices.end(), 0);

      if(!splitterIndices.empty()) {
        auto newStages = stage.split(splitterIndices);
        stageIt = multiStage->childrenErase(stageIt);
        stageIt = multiStage->insertChildren(stageIt, std::make_move_iterator(newStages.begin()),
                                             std::make_move_iterator(newStages.end()));
        std::advance(stageIt, newStages.size() - 1);
      }
    }
  }

  return true;
}

} // namespace dawn
