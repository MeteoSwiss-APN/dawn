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
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/Support/Logging.h"
#include <vector>

namespace dawn {

bool PassManager::runAllPassesOnStecilInstantiation(
    const std::shared_ptr<StencilInstantiation>& instantiation) {
  std::vector<std::string> passesRan;

  for(auto& pass : passes_) {
    for(const auto& dependency : pass->getDependencies())
      if(std::find(passesRan.begin(), passesRan.end(), dependency) == passesRan.end()) {
        DiagnosticsBuilder diag(DiagnosticsKind::Error);
        diag << "invalid pass registration: optimizer pass '" << pass->getName() << "' depends on '"
             << dependency << "'";
        instantiation->getOptimizerContext()->getDiagnostics().report(diag);
        return false;
      }

    if(!runPassOnStecilInstantiation(instantiation, pass.get()))
      return false;

    passesRan.emplace_back(pass->getName());
  }
  return true;
}

bool PassManager::runPassOnStecilInstantiation(
    const std::shared_ptr<StencilInstantiation>& instantiation, Pass* pass) {
  DAWN_LOG(INFO) << "Starting " << pass->getName() << " ...";

  if(!pass->run(instantiation)) {
    DAWN_LOG(WARNING) << "Done with " << pass->getName() << " : FAIL";
    return false;
  }

  DAWN_LOG(INFO) << "Done with " << pass->getName() << " : Success";
  return true;
}

} // namespace dawn
