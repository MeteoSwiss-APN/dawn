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

#include "gtclang/Frontend/GTClangASTConsumer.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Config.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/StringUtil.h"
#include "dawn/Support/Unreachable.h"
#include "gtclang/Frontend/ClangFormat.h"
#include "gtclang/Frontend/GTClangASTAction.h"
#include "gtclang/Frontend/GTClangASTVisitor.h"
#include "gtclang/Frontend/GTClangContext.h"
#include "gtclang/Frontend/GlobalVariableParser.h"
#include "gtclang/Frontend/StencilParser.h"
#include "gtclang/Support/ClangCompat/FileSystem.h"
#include "gtclang/Support/ClangCompat/VirtualFileSystem.h"
#include "gtclang/Support/Config.h"
#include "gtclang/Support/FileUtil.h"
#include "gtclang/Support/Logger.h"
#include "gtclang/Support/StringUtil.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Basic/Version.h"
#include "llvm/ADT/StringSwitch.h"

namespace gtclang {

GTClangASTConsumer::GTClangASTConsumer(GTClangContext* context, const std::string& file,
                                       GTClangASTAction* parentAction)
    : context_(context), file_(file), parentAction_(parentAction) {
  DAWN_LOG(INFO) << "Creating ASTVisitor ... ";
  visitor_ = std::make_unique<GTClangASTVisitor>(context);
}

void GTClangASTConsumer::HandleTranslationUnit(clang::ASTContext& ASTContext) {
  context_->setASTContext(&ASTContext);
  if(!context_->hasDiagnostics())
    context_->setDiagnostics(&ASTContext.getDiagnostics());

  DAWN_LOG(INFO) << "Parsing translation unit... ";

  clang::TranslationUnitDecl* TU = ASTContext.getTranslationUnitDecl();

  if(context_->getOptions().DumpAST) {
    TU->dumpColor();
  }

  for(auto& decl : TU->decls())
    visitor_->TraverseDecl(decl);

  DAWN_LOG(INFO) << "Done parsing translation unit";

  if(context_->getASTContext().getDiagnostics().hasErrorOccurred()) {
    DAWN_LOG(INFO) << "Erros occurred. Aborting";
    return;
  }

  DAWN_LOG(INFO) << "Generating SIR ... ";
  auto& SM = context_->getSourceManager();

  // Assemble SIR
  std::shared_ptr<dawn::SIR> SIR = std::make_shared<dawn::SIR>(dawn::ast::GridType::Cartesian);
  SIR->Filename = SM.getFileEntryForID(SM.getMainFileID())->getName();

  const StencilParser& stencilParser = visitor_->getStencilParser();

  for(const auto& stencilPair : stencilParser.getStencilMap()) {
    SIR->Stencils.emplace_back(stencilPair.second);
    SIR->Stencils.back()->Attributes = context_->getStencilAttribute(stencilPair.second->Name);
  }

  for(const auto& stencilPair : stencilParser.getStencilFunctionMap()) {
    SIR->StencilFunctions.emplace_back(stencilPair.second);
    SIR->StencilFunctions.back()->Attributes =
        context_->getStencilAttribute(stencilPair.second->Name);
  }

  const GlobalVariableParser& globalsParser = visitor_->getGlobalVariableParser();
  SIR->GlobalVariableMap = globalsParser.getGlobalVariableMap();

  if(context_->getOptions().DumpSIR) {
    SIR->dump();
  }

  parentAction_->setSIR(SIR);
}

} // namespace gtclang
