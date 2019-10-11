//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _       _
//                        | |     | |
//                    __ _| |_ ___| | __ _ _ __   __ _
//                   / _` | __/ __| |/ _` | '_ \ / _` |
//                  | (_| | || (__| | (_| | | | | (_| |
//                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
//                    __/ |                       __/ |
//                   |___/                       |___/
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gtclang/Frontend/GTClangContext.h"
#include "dawn/Support/Assert.h"
#include "clang/AST/ASTContext.h"
#include "llvm/ADT/STLExtras.h"

namespace gtclang {

GTClangContext::GTClangContext()
    : options_(std::make_unique<Options>()), diagnostics_(nullptr), astContext_(nullptr) {}

Options& GTClangContext::getOptions() {
  DAWN_ASSERT(options_);
  return *options_;
}

const Options& GTClangContext::getOptions() const {
  DAWN_ASSERT(options_);
  return *options_;
}

Diagnostics& GTClangContext::getDiagnostics() {
  DAWN_ASSERT(diagnostics_);
  return *diagnostics_;
}

const Diagnostics& GTClangContext::getDiagnostics() const {
  DAWN_ASSERT(diagnostics_);
  return *diagnostics_;
}

void GTClangContext::setDiagnostics(clang::DiagnosticsEngine* diags) {
  DAWN_ASSERT_MSG(!diagnostics_, "Diagnostics already set!");
  diagnostics_ = std::make_unique<Diagnostics>(diags);
}

bool GTClangContext::hasDiagnostics() const { return (diagnostics_ != nullptr); }

clang::ASTContext& GTClangContext::getASTContext() {
  DAWN_ASSERT(astContext_);
  return *astContext_;
}

const clang::ASTContext& GTClangContext::getASTContext() const {
  DAWN_ASSERT(astContext_);
  return *astContext_;
}

void GTClangContext::setASTContext(clang::ASTContext* astContext) {
  DAWN_ASSERT_MSG(!astContext_, "AST Context already set!");
  astContext_ = astContext;
}

clang::SourceManager& GTClangContext::getSourceManager() {
  DAWN_ASSERT(astContext_);
  return astContext_->getSourceManager();
}

const clang::SourceManager& GTClangContext::getSourceManager() const {
  DAWN_ASSERT(astContext_);
  return astContext_->getSourceManager();
}

clang::DiagnosticsEngine& GTClangContext::getDiagnosticsEngine() {
  DAWN_ASSERT(diagnostics_);
  return diagnostics_->getDiagnosticsEngine();
}

const clang::DiagnosticsEngine& GTClangContext::getDiagnosticsEngine() const {
  DAWN_ASSERT(diagnostics_);
  return diagnostics_->getDiagnosticsEngine();
}

dawn::sir::Attr GTClangContext::getStencilAttribute(const std::string& name) const {
  auto it = stencilNameToAttributeMap_.find(name);
  return it != stencilNameToAttributeMap_.end() ? it->second : dawn::sir::Attr();
}

void GTClangContext::setStencilAttribute(const std::string& name, dawn::sir::Attr attr) {
  stencilNameToAttributeMap_.emplace(name, attr);
}

} // namespace gtclang
