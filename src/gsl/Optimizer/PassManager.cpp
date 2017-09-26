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

#include "gsl/Optimizer/PassManager.h"
#include "gsl/Optimizer/OptimizerContext.h"
#include "gsl/Optimizer/StencilInstantiation.h"
#include "gsl/Support/Logging.h"
#include <vector>

namespace gsl {

bool PassManager::runAllPassesOnStecilInstantiation(StencilInstantiation* instantiation) {
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

bool PassManager::runPassOnStecilInstantiation(StencilInstantiation* instantiation, Pass* pass) {
  GSL_LOG(INFO) << "Starting " << pass->getName() << " ...";

  if(!pass->run(instantiation)) {
    GSL_LOG(WARNING) << "Done with " << pass->getName() << " : FAIL";
    return false;
  }

  GSL_LOG(INFO) << "Done with " << pass->getName() << " : Success";
  return true;
}

} // namespace gsl
