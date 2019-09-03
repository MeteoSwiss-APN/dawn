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

#include "dawn/IIR/StencilFunction.h"
#include "dawn/IIR/AST.h"
#include "dawn/IIR/SIRToIIRASTMapper.h"
#include "dawn/SIR/AST.h"
#include "dawn/SIR/SIR.h"

namespace dawn {

namespace iir {

StencilFunction::StencilFunction(const sir::StencilFunction& sirSF)
    : Name(sirSF.Name), Args(sirSF.Args), Intervals(sirSF.Intervals) {
  for(const auto& ast : sirSF.Asts) {
    SIRToIIRASTMapper mapper;
    ast->accept(mapper);
    Asts.push_back(std::make_shared<iir::AST>(
        std::dynamic_pointer_cast<iir::BlockStmt>(mapper.getStmtMap().at(ast->getRoot()))));
  }
}

std::shared_ptr<iir::AST> StencilFunction::getASTOfInterval(const sir::Interval& interval) const {
  for(int i = 0; i < Intervals.size(); ++i)
    if(*Intervals[i] == interval)
      return Asts[i];
  return nullptr;
}

bool StencilFunction::hasArg(std::string name) {
  return std::find_if(Args.begin(), Args.end(), [&](std::shared_ptr<sir::StencilFunctionArg> arg) {
           return name == arg->Name;
         }) != Args.end();
}

} // namespace iir
} // namespace dawn
