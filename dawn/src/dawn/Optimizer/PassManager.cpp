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

#include "dawn/Optimizer/PassManager.h"
#include "dawn/AST/GridType.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Support/Exception.h"
#include "dawn/Support/Logger.h"
#include <vector>

namespace dawn {

bool PassManager::runAllPassesOnStencilInstantiation(
    OptimizerContext& context, const std::shared_ptr<iir::StencilInstantiation>& instantiation) {
  std::vector<std::string> passesRan;

  for(auto& pass : passes_) {
    for(const auto& dependency : pass->getDependencies())
      if(std::find(passesRan.begin(), passesRan.end(), dependency) == passesRan.end()) {
        throw LogicError(std::string("Invalid pass registration: optimizer pass '") +
                         pass->getName() + "' depends on '" + dependency + "'");
      }

    if(!runPassOnStencilInstantiation(context, instantiation, pass.get()))
      return false;

    passesRan.emplace_back(pass->getName());
  }
  return true;
}

bool PassManager::runPassOnStencilInstantiation(
    OptimizerContext& context, const std::shared_ptr<iir::StencilInstantiation>& instantiation,
    Pass* pass) {
  DAWN_LOG(INFO) << "Starting " << pass->getName() << " ...";

  if(!pass->run(instantiation)) {
    DAWN_LOG(WARNING) << "Done with " << pass->getName() << " : FAIL";
    return false;
  }

  if(context.getOptions().PassVerbose) {
    instantiation->jsonDump(pass->getName() + "_" + std::to_string(passCounter_[pass->getName()]) +
                            "_Log.json");
  }

  passCounter_[pass->getName()]++;
  DAWN_LOG(INFO) << "Done with " << pass->getName() << " : Success";
  return true;
}

} // namespace dawn
