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

#include "gtclang/Frontend/GTClangASTAction.h"
#include "gtclang/Frontend/GTClangASTConsumer.h"
#include "gtclang/Support/Logger.h"
#include "clang/Frontend/CompilerInstance.h"

namespace gtclang {

GTClangASTAction::GTClangASTAction(GTClangContext* context) : context_(context) {}

std::unique_ptr<clang::ASTConsumer>
GTClangASTAction::CreateASTConsumer(clang::CompilerInstance& compiler, llvm::StringRef file) {
  DAWN_LOG(INFO) << "Creating ASTConsumer for " << file.str();
  return std::make_unique<GTClangASTConsumer>(context_, file, this);
}

void GTClangASTAction::setSIR(std::shared_ptr<dawn::SIR> sir) { sir_ = sir; }

std::shared_ptr<dawn::SIR> GTClangASTAction::getSIR() const { return sir_; }

} // namespace gtclang
