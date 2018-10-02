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

#include "dawn/IIR/StencilFunction/FunctionHandeling.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/MetaInformation.h"
#include "dawn/IIR/StencilFunction/StencilFunctionInstantiation.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/RemoveIf.hpp"

namespace dawn {
namespace iir {
namespace StencilFunctionHandeling {

void deregisterStencilFunction(std::shared_ptr<dawn::iir::StencilFunctionInstantiation> stencilFun,
                               dawn::iir::IIR* iir) {
  bool found = RemoveIf(iir->getMetaData()->getExprToStencilFunctionInstantiationMap(),
                        [&](std::pair<std::shared_ptr<StencilFunCallExpr>,
                                      std::shared_ptr<StencilFunctionInstantiation>>
                                pair) { return (pair.second == stencilFun); });
  // make sure the element existed and was removed
  DAWN_ASSERT(found);
  found = RemoveIf(
      iir->getMetaData()->getStencilFunctionInstantiations(),
      [&](const std::shared_ptr<StencilFunctionInstantiation>& v) { return (v == stencilFun); });

  // make sure the element existed and was removed
  DAWN_ASSERT(found);
}

void registerStencilFunction(std::shared_ptr<StencilFunctionInstantiation> stencilFun, IIR* iir) {
  // TODO gather all the stencil function properties in a struct
  //  DAWN_ASSERT(ExprToStencilFunctionInstantiationMap_.count(stencilFun->getExpression()));
  //  DAWN_ASSERT(nameToStencilFunctionInstantiationMap_.count(stencilFun->getExpression()));
  iir->getMetaData()->getExprToStencilFunctionInstantiationMap().emplace(
      stencilFun->getExpression(), stencilFun);
}

void finalizeStencilFunctionSetup(std::shared_ptr<StencilFunctionInstantiation> stencilFun,
                                  IIR* iir) {
  DAWN_ASSERT(iir->getMetaData()->getStencilFunInstantiationCandidate().count(stencilFun));
  stencilFun->closeFunctionBindings();
  // We take the candidate to stencil function and placed it in the stencil function instantiations
  // container
  iir::StencilMetaInformation::StencilFunctionInstantiationCandidate candidate =
      iir->getMetaData()->getStencilFunInstantiationCandidate()[stencilFun];

  // map of expr to stencil function instantiation is updated
  if(candidate.callerStencilFunction_) {
    candidate.callerStencilFunction_->insertExprToStencilFunction(stencilFun);
  } else {
    registerStencilFunction(stencilFun, iir);
  }

  stencilFun->update();

  iir->getMetaData()->getStencilFunctionInstantiations().push_back(stencilFun);
  // we remove the candidate to stencil function
  iir->getMetaData()->getStencilFunInstantiationCandidate().erase(stencilFun);
}

} // namespace StencilFunctionHandeling
} // namespace iir
} // namespace dawn
