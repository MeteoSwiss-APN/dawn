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

#include "gsl/Optimizer/OptimizerContext.h"
#include "gsl/Compiler/GSLCompiler.h"
#include "gsl/Optimizer/StencilInstantiation.h"
#include "gsl/SIR/SIR.h"
#include "gsl/Support/Logging.h"
#include "gsl/Support/STLExtras.h"

namespace gsl {

OptimizerContext::OptimizerContext(GSLCompiler* compiler, const SIR* SIR)
    : compiler_(compiler), SIR_(SIR) {
  GSL_LOG(INFO) << "Intializing OptimizerContext ... ";

  for(const auto& stencil : SIR_->Stencils)
    if(!stencil->Attributes.has(sir::Attr::AK_NoCodeGen)) {
      stencilInstantiationMap_.insert(std::make_pair(
          stencil->Name, make_unique<StencilInstantiation>(this, stencil.get(), SIR)));
    } else {
      GSL_LOG(INFO) << "Skipping processing of `" << stencil->Name << "`";
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

} // namespace gsl
