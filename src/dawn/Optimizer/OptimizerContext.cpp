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

OptimizerContext::OptimizerContext(DiagnosticsEngine& diagnostics, Options& options,
                                   const std::shared_ptr<SIR>& SIR)
    : diagnostics_(diagnostics), options_(options), SIR_(SIR) {
  DAWN_LOG(INFO) << "Intializing OptimizerContext ... ";

  for(const auto& stencil : SIR_->Stencils)
    if(!stencil->Attributes.has(sir::Attr::AK_NoCodeGen)) {
      stencilInstantiationMap_.insert(std::make_pair(
          stencil->Name, std::make_shared<StencilInstantiation>(this, stencil, SIR)));
    } else {
      DAWN_LOG(INFO) << "Skipping processing of `" << stencil->Name << "`";
    }
}

std::map<std::string, std::shared_ptr<StencilInstantiation>>&
OptimizerContext::getStencilInstantiationMap() {
  return stencilInstantiationMap_;
}

const std::map<std::string, std::shared_ptr<StencilInstantiation>>&
OptimizerContext::getStencilInstantiationMap() const {
  return stencilInstantiationMap_;
}

const DiagnosticsEngine& OptimizerContext::getDiagnostics() const { return diagnostics_; }

DiagnosticsEngine& OptimizerContext::getDiagnostics() { return diagnostics_; }

const Options& OptimizerContext::getOptions() const { return options_; }

Options& OptimizerContext::getOptions() { return options_; }

} // namespace dawn
