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

#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/STLExtras.h"

namespace dawn {

OptimizerContext::OptimizerContext(DawnCompiler* compiler, const SIR* SIR)
    : compiler_(compiler), SIR_(SIR) {
  DAWN_LOG(INFO) << "Intializing OptimizerContext ... ";

  for(const auto& stencil : SIR_->Stencils)
    if(!stencil->Attributes.has(sir::Attr::AK_NoCodeGen)) {
      stencilInstantiationMap_.insert(std::make_pair(
          stencil->Name, make_unique<StencilInstantiation>(this, stencil.get(), SIR)));
    } else {
      DAWN_LOG(INFO) << "Skipping processing of `" << stencil->Name << "`";
    }
}

std::map<std::string, std::unique_ptr<StencilInstantiation>>&
OptimizerContext::getStencilInstantiationMap() {
  return stencilInstantiationMap_;
}

const std::map<std::string, std::unique_ptr<StencilInstantiation>>&
OptimizerContext::getStencilInstantiationMap() const {
  return stencilInstantiationMap_;
}

const DiagnosticsEngine& OptimizerContext::getDiagnostics() const {
  return compiler_->getDiagnostics();
}

DiagnosticsEngine& OptimizerContext::getDiagnostics() { return compiler_->getDiagnostics(); }

const Options& OptimizerContext::getOptions() const { return compiler_->getOptions(); }

Options& OptimizerContext::getOptions() { return compiler_->getOptions(); }

} // namespace dawn
