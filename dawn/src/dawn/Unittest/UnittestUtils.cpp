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

#include "UnittestUtils.h"
#include "dawn/IIR/IIRNodeIterator.h"

namespace dawn {
iir::DoMethod& getFirstDoMethod(std::shared_ptr<iir::StencilInstantiation>& si) {
  return **iterateIIROver<iir::DoMethod>(*si->getIIR()).begin();
}

std::shared_ptr<iir::Stmt> getNthStmt(iir::DoMethod& doMethod, int n) {
  return doMethod.getAST().getStatements()[n];
}
} // namespace dawn