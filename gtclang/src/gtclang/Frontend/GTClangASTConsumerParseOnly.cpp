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

#include "dawn/AST/GridType.h"
#include "dawn/CodeGen/TranslationUnit.h"
#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/Compiler/Options.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "dawn/Support/FileSystem.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/Logging.h"
#include "gtclang/Frontend/ClangFormat.h"
#include "gtclang/Frontend/Diagnostics.h"
#include "gtclang/Frontend/GTClangASTAction.h"
#include "gtclang/Frontend/GTClangASTConsumerParseOnly.h"
#include "gtclang/Frontend/GTClangASTVisitor.h"
#include "gtclang/Frontend/GTClangContext.h"
#include "gtclang/Support/ClangCompat/FileSystem.h"
#include "gtclang/Support/ClangCompat/VirtualFileSystem.h"
#include "gtclang/Support/Config.h"
#include "gtclang/Support/FileUtil.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include <ctime>
#include <iostream>
#include <memory>
#include <string>

namespace gtclang {

GTClangASTConsumerParseOnly::GTClangASTConsumerParseOnly(GTClangContext* context,
                                                         const std::string& file,
                                                         GTClangASTAction* parentAction)
    : context_(context), file_(file), parentAction_(parentAction) {
  DAWN_LOG(INFO) << "Creating ASTVisitor ... ";
  visitor_ = std::make_unique<GTClangASTVisitor>(context);
}

void GTClangASTConsumerParseOnly::HandleTranslationUnit(clang::ASTContext& ASTContext) {
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

  // Determine filename of generated file (by default we append "_gen" to the filename)
  std::string generatedFileName;
  if(context_->getOptions().OutputFile.empty())
    generatedFileName = std::string(fs::path(file_).filename().stem()) + "_gen.cpp";
  else {
    generatedFileName = context_->getOptions().OutputFile;
  }

  if(context_->getOptions().WriteSIR) {
    const std::string generatedSIR =
        std::string(fs::path(generatedFileName).filename().stem()) + ".sir";
    DAWN_LOG(INFO) << "Generating SIR file " << generatedSIR;

    if(context_->getOptions().SIRFormat == "json") {
      dawn::SIRSerializer::serialize(generatedSIR, SIR.get(), dawn::SIRSerializer::Format::Json);
    } else if(context_->getOptions().SIRFormat == "byte") {
      dawn::SIRSerializer::serialize(generatedSIR, SIR.get(), dawn::SIRSerializer::Format::Byte);

    } else {
      dawn_unreachable("Unknown SIRFormat option");
    }
  }
}

} // namespace gtclang
